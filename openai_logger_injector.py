
#!/usr/bin/env python3
"""
OpenAI Logger Injector

This script automatically analyzes Python codebases to find OpenAI API calls
and injects appropriate logging code. It can be used to standardize logging
across projects that use the OpenAI API.

Usage:
    python openai_logger_injector.py [--dry-run] [--backup] [--report] path/to/project

Options:
    --dry-run       Only detect calls without modifying files
    --backup        Create backups of modified files
    --report        Generate a detailed report of findings and changes
    --verbose       Show detailed information during processing
"""

import ast
import os
import re
import sys
import argparse
import shutil
import json
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime

# ANSI color codes for terminal output
COLORS = {
    "RESET": "\033[0m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "BOLD": "\033[1m",
}

@dataclass
class OpenAICall:
    """Represents a detected OpenAI API call."""
    file_path: str
    line_number: int
    column: int
    function_name: str
    method_name: str  # The specific API method (e.g., create)
    is_streaming: bool = False
    parent_function: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    implementation_status: str = "Not Implemented"
    
    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_number} - {self.function_name}.{self.method_name}()"

@dataclass
class FileReport:
    """Report for a single file."""
    file_path: str
    calls_detected: List[OpenAICall] = field(default_factory=list)
    has_logger_import: bool = False
    needs_modification: bool = False
    was_modified: bool = False
    error: Optional[str] = None

@dataclass
class ProjectReport:
    """Overall report for the project."""
    project_path: str
    files_analyzed: int = 0
    files_with_openai_calls: int = 0
    files_modified: int = 0
    total_calls_detected: int = 0
    total_calls_logged: int = 0
    file_reports: List[FileReport] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class OpenAICallVisitor(ast.NodeVisitor):
    """AST visitor to detect OpenAI API calls."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.calls = []
        self.current_function = None
        self.current_class = None
        self.openai_client_vars = set()  # Variables that hold OpenAI client instances
        self.instance_openai_clients = set()  # Instance variables that hold OpenAI clients (e.g., self.openai_client)
        
    def visit_ClassDef(self, node):
        """Visit class definitions to track context."""
        old_class = self.current_class
        old_function = self.current_function
        self.current_class = node.name
        self.current_function = None
        self.generic_visit(node)
        self.current_class = old_class
        self.current_function = old_function
        
    def visit_FunctionDef(self, node):
        """Visit function definitions to track context."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
        
    def visit_Assign(self, node):
        """Visit assignments to detect OpenAI client initialization."""
        # Check for OpenAI client initialization
        if isinstance(node.value, ast.Call):
            func = node.value.func
            is_openai_client = False
            
            # Check for AzureOpenAI or openai.OpenAI initialization
            if isinstance(func, ast.Name) and func.id == 'AzureOpenAI':
                is_openai_client = True
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == 'openai' and func.attr == 'OpenAI':
                    is_openai_client = True
            
            if is_openai_client:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.openai_client_vars.add(target.id)
                    elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == 'self':
                            # This is a class instance variable (e.g., self.openai_client)
                            self.instance_openai_clients.add(target.attr)
        
        self.generic_visit(node)
        
    def _is_openai_api_call(self, node):
        """Check if a node represents an OpenAI API call."""
        if not isinstance(node.func, ast.Attribute):
            return False, None, None
        
        # Check for patterns like client.chat.completions.create() or client.embeddings.create()
        if isinstance(node.func.value, ast.Attribute):
            # Pattern: client.service.method() or client.service.subservice.method()
            
            # Check for client.service.method() pattern
            if isinstance(node.func.value.value, ast.Name):
                # Regular variable: client.service.method()
                if node.func.value.value.id in self.openai_client_vars:
                    return True, node.func.value.value.id, node.func.value.attr
            
            # Check for self.client.service.method() pattern
            elif isinstance(node.func.value.value, ast.Attribute) and isinstance(node.func.value.value.value, ast.Name):
                # Instance variable: self.client.service.method()
                if (node.func.value.value.value.id == 'self' and 
                    node.func.value.value.attr in self.instance_openai_clients):
                    return True, f"self.{node.func.value.value.attr}", node.func.value.attr
            
            # Check for client.service.subservice.method() pattern (e.g., client.chat.completions.create())
            elif isinstance(node.func.value, ast.Attribute) and isinstance(node.func.value.value, ast.Attribute):
                if isinstance(node.func.value.value.value, ast.Name):
                    # Regular variable: client.service.subservice.method()
                    if node.func.value.value.value.id in self.openai_client_vars:
                        service = f"{node.func.value.value.attr}.{node.func.value.attr}"
                        return True, node.func.value.value.value.id, service
                
                # Check for self.client.service.subservice.method() pattern
                elif (isinstance(node.func.value.value.value, ast.Attribute) and 
                      isinstance(node.func.value.value.value.value, ast.Name) and
                      node.func.value.value.value.value.id == 'self'):
                    # Instance variable: self.client.service.subservice.method()
                    client_attr = node.func.value.value.value.attr
                    if client_attr in self.instance_openai_clients:
                        service = f"{node.func.value.value.attr}.{node.func.value.attr}"
                        return True, f"self.{client_attr}", service
        
        # Check for direct patterns like client.create() or self.client.create()
        elif isinstance(node.func.value, ast.Name):
            if node.func.value.id in self.openai_client_vars and node.func.attr == 'create':
                return True, node.func.value.id, None
        elif isinstance(node.func.value, ast.Attribute) and isinstance(node.func.value.value, ast.Name):
            if (node.func.value.value.id == 'self' and 
                node.func.value.attr in self.instance_openai_clients and 
                node.func.attr == 'create'):
                return True, f"self.{node.func.value.attr}", None
        
        return False, None, None
        
    def visit_Call(self, node):
        """Visit function calls to detect OpenAI API calls."""
        is_openai_call, client_var, service = self._is_openai_api_call(node)
        
        if is_openai_call:
            method = node.func.attr  # e.g., 'create'
            
            # Determine if this is a streaming call
            is_streaming = False
            for keyword in node.keywords:
                if keyword.arg == 'stream' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    is_streaming = True
                    break
            
            # Extract arguments
            args = {}
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Constant):
                    args[keyword.arg] = keyword.value.value
                elif isinstance(keyword.value, ast.Name):
                    args[keyword.arg] = keyword.value.id
                else:
                    args[keyword.arg] = "complex_expression"
            
            function_name = client_var
            if service:
                function_name = f"{client_var}.{service}"
            
            call = OpenAICall(
                file_path=self.file_path,
                line_number=node.lineno,
                column=node.col_offset,
                function_name=function_name,
                method_name=method,
                is_streaming=is_streaming,
                parent_function=self.current_function,
                args=args
            )
            self.calls.append(call)
        
        self.generic_visit(node)

def analyze_file(file_path: str) -> FileReport:
    """
    Analyze a Python file to detect OpenAI API calls.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        FileReport object with analysis results
    """
    report = FileReport(file_path=file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file already imports the logger
        report.has_logger_import = bool(re.search(r'from\s+openai_logger\s+import\s+log_openai_call', content))
        
        # Parse the file
        tree = ast.parse(content)
        visitor = OpenAICallVisitor(file_path)
        visitor.visit(tree)
        
        report.calls_detected = visitor.calls
        report.needs_modification = bool(visitor.calls) and not report.has_logger_import
        
        return report
    except Exception as e:
        report.error = str(e)
        return report

def backup_file(file_path: str) -> str:
    """
    Create a backup of a file before modifying it.
    
    Args:
        file_path: Path to the file to backup
        
    Returns:
        Path to the backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.{timestamp}.bak"
    shutil.copy2(file_path, backup_path)
    return backup_path

def inject_logging(file_path: str, report: FileReport) -> bool:
    """
    Inject logging code into a file.
    
    Args:
        file_path: Path to the file to modify
        report: FileReport with analysis results
        
    Returns:
        True if the file was modified, False otherwise
    """
    if not report.calls_detected:
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Add import if needed
        if not report.has_logger_import:
            # Find a good place to add the import
            import_line = "from openai_logger import log_openai_call\n"
            
            # Look for other imports to add after
            import_added = False
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    # Find the last import line
                    last_import_line = i
                    for j in range(i + 1, len(lines)):
                        if lines[j].startswith('import ') or lines[j].startswith('from '):
                            last_import_line = j
                        elif not lines[j].strip() or lines[j].startswith('#'):
                            continue
                        else:
                            break
                    
                    # Add after the last import
                    lines.insert(last_import_line + 1, import_line)
                    import_added = True
                    break
            
            # If no imports found, add at the top
            if not import_added:
                lines.insert(0, import_line)
        
        # Process calls in reverse order to avoid line number changes
        calls = sorted(report.calls_detected, key=lambda c: c.line_number, reverse=True)
        
        for call in calls:
            line_idx = call.line_number - 1
            
            # Find the full call expression which might span multiple lines
            call_start_line = line_idx
            call_text = lines[line_idx]
            
            # Check if the call spans multiple lines
            open_parens = call_text.count('(') - call_text.count(')')
            while open_parens > 0 and call_start_line + 1 < len(lines):
                call_start_line += 1
                next_line = lines[call_start_line]
                call_text += next_line
                open_parens += next_line.count('(') - next_line.count(')')
            
            # Determine indentation
            indent = re.match(r'^(\s*)', lines[line_idx]).group(1)
            
            # Create the logging code
            if call.is_streaming:
                # For streaming calls
                logging_code = [
                    f"{indent}request = {{\n",
                    f"{indent}    # Arguments for {call.function_name}.{call.method_name}\n"
                ]
                
                # Add known arguments
                for arg, value in call.args.items():
                    if isinstance(value, str) and value == "complex_expression":
                        # For complex expressions, add a comment
                        logging_code.append(f"{indent}    # '{arg}': <complex expression>,\n")
                    else:
                        # For simple values
                        if isinstance(value, str):
                            logging_code.append(f"{indent}    '{arg}': {value},\n")
                        else:
                            logging_code.append(f"{indent}    '{arg}': {repr(value)},\n")
                
                logging_code.extend([
                    f"{indent}}}\n",
                    f"{indent}log_openai_call(request, {{\"type\": \"stream_started\"}})\n"
                ])
                
                # Find the line with the actual API call
                api_call_pattern = re.compile(rf'{re.escape(call.function_name)}\.{call.method_name}\s*\(')
                
                # Insert the logging code before the API call
                for i in range(line_idx, call_start_line + 1):
                    if api_call_pattern.search(lines[i]):
                        # Replace the direct call with one using **request
                        original_call = lines[i]
                        modified_call = api_call_pattern.sub(f'{call.function_name}.{call.method_name}(**request', original_call)
                        lines[i] = modified_call
                        
                        # Insert the logging code before this line
                        for log_line in reversed(logging_code):
                            lines.insert(i, log_line)
                        break
            else:
                # For non-streaming calls
                logging_code = [
                    f"{indent}request = {{\n",
                    f"{indent}    # Arguments for {call.function_name}.{call.method_name}\n"
                ]
                
                # Add known arguments
                for arg, value in call.args.items():
                    if isinstance(value, str) and value == "complex_expression":
                        # For complex expressions, add a comment
                        logging_code.append(f"{indent}    # '{arg}': <complex expression>,\n")
                    else:
                        # For simple values
                        if isinstance(value, str):
                            logging_code.append(f"{indent}    '{arg}': {value},\n")
                        else:
                            logging_code.append(f"{indent}    '{arg}': {repr(value)},\n")
                
                logging_code.append(f"{indent}}}\n")
                
                # Find the line with the actual API call
                api_call_pattern = re.compile(rf'{re.escape(call.function_name)}\.{call.method_name}\s*\(')
                
                # Insert the logging code before the API call and modify the call
                for i in range(line_idx, call_start_line + 1):
                    if api_call_pattern.search(lines[i]):
                        # Check if the result is assigned to a variable
                        assignment_match = re.match(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*', lines[i])
                        
                        if assignment_match:
                            # There's an assignment, modify it
                            var_name = assignment_match.group(2)
                            modified_call = api_call_pattern.sub(f'{call.function_name}.{call.method_name}(**request', lines[i])
                            lines[i] = modified_call
                            
                            # Add logging after the call
                            log_after = f"{indent}log_openai_call(request, {var_name})\n"
                            lines.insert(i + 1, log_after)
                        else:
                            # No assignment, just a call
                            modified_call = api_call_pattern.sub(f'response = {call.function_name}.{call.method_name}(**request', lines[i])
                            lines[i] = modified_call
                            
                            # Add logging after the call
                            log_after = f"{indent}log_openai_call(request, response)\n"
                            lines.insert(i + 1, log_after)
                        
                        # Insert the request code before this line
                        for log_line in reversed(logging_code):
                            lines.insert(i, log_line)
                        break
        
        # Write the modified file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return True
    except Exception as e:
        print(f"Error modifying {file_path}: {e}")
        return False

def generate_report_table(project_report: ProjectReport) -> str:
    """
    Generate a markdown table summarizing the OpenAI calls.
    
    Args:
        project_report: ProjectReport with analysis results
        
    Returns:
        Markdown table as a string
    """
    table = "# OpenAI Logger Implementation Report\n\n"
    table += f"Project: {project_report.project_path}\n"
    table += f"Generated: {project_report.timestamp}\n\n"
    table += f"Files analyzed: {project_report.files_analyzed}\n"
    table += f"Files with OpenAI calls: {project_report.files_with_openai_calls}\n"
    table += f"Files modified: {project_report.files_modified}\n"
    table += f"Total calls detected: {project_report.total_calls_detected}\n"
    table += f"Total calls logged: {project_report.total_calls_logged}\n\n"
    
    table += "## OpenAI API Calls\n\n"
    table += "| API Call | File | Function | Line | Implementation Status |\n"
    table += "|----------|------|----------|------|------------------------|\n"
    
    # Sort calls by file and line number
    all_calls = []
    for report in project_report.file_reports:
        all_calls.extend(report.calls_detected)
    
    all_calls.sort(key=lambda c: (c.file_path, c.line_number))
    
    for call in all_calls:
        file_name = os.path.basename(call.file_path)
        function_name = call.parent_function or "global"
        table += f"| `{call.function_name}.{call.method_name}` | {file_name} | {function_name} | {call.line_number} | {call.implementation_status} |\n"
    
    return table

def process_project(project_path: str, dry_run: bool = False, backup: bool = False, 
                   report: bool = False, verbose: bool = False) -> ProjectReport:
    """
    Process a project to detect and inject OpenAI logging.
    
    Args:
        project_path: Path to the project directory
        dry_run: If True, only detect calls without modifying files
        backup: If True, create backups of modified files
        report: If True, generate a detailed report
        verbose: If True, show detailed information during processing
        
    Returns:
        ProjectReport with analysis results
    """
    project_report = ProjectReport(project_path=project_path)
    
    # Directories to skip
    skip_dirs = ['.venv', 'venv', 'env', '__pycache__', 'node_modules']
    
    # Walk through the project directory
    for root, dirs, files in os.walk(project_path):
        # Skip directories in the skip_dirs list
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                if verbose:
                    print(f"Analyzing {file_path}...")
                
                # Analyze the file
                file_report = analyze_file(file_path)
                project_report.files_analyzed += 1
                
                if file_report.error:
                    if verbose:
                        print(f"{COLORS['RED']}Error analyzing {file_path}: {file_report.error}{COLORS['RESET']}")
                    continue
                
                if file_report.calls_detected:
                    project_report.files_with_openai_calls += 1
                    project_report.total_calls_detected += len(file_report.calls_detected)
                    
                    if verbose:
                        print(f"{COLORS['GREEN']}Found {len(file_report.calls_detected)} OpenAI calls in {file_path}{COLORS['RESET']}")
                        for call in file_report.calls_detected:
                            print(f"  Line {call.line_number}: {call.function_name}.{call.method_name}()")
                    
                    # Modify the file if needed
                    if not dry_run and file_report.needs_modification:
                        if backup:
                            backup_path = backup_file(file_path)
                            if verbose:
                                print(f"Created backup: {backup_path}")
                        
                        modified = inject_logging(file_path, file_report)
                        file_report.was_modified = modified
                        
                        if modified:
                            project_report.files_modified += 1
                            project_report.total_calls_logged += len(file_report.calls_detected)
                            
                            # Update implementation status
                            for call in file_report.calls_detected:
                                call.implementation_status = "Implemented"
                            
                            if verbose:
                                print(f"{COLORS['BLUE']}Modified {file_path}{COLORS['RESET']}")
                
                project_report.file_reports.append(file_report)
    
    # Generate report if requested
    if report:
        report_text = generate_report_table(project_report)
        report_path = os.path.join(project_path, "openai_logger_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        if verbose:
            print(f"{COLORS['GREEN']}Report generated: {report_path}{COLORS['RESET']}")
    
    return project_report

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Inject OpenAI logging into Python codebase')
    parser.add_argument('path', help='Path to the project directory')
    parser.add_argument('--dry-run', action='store_true', help='Only detect calls without modifying files')
    parser.add_argument('--backup', action='store_true', help='Create backups of modified files')
    parser.add_argument('--report', action='store_true', help='Generate a detailed report')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information during processing')
    args = parser.parse_args()
    
    print(f"{COLORS['BOLD']}OpenAI Logger Injector{COLORS['RESET']}")
    print(f"Analyzing project: {args.path}")
    print(f"Dry run: {args.dry_run}")
    print(f"Create backups: {args.backup}")
    print(f"Generate report: {args.report}")
    print(f"Verbose output: {args.verbose}")
    print()
    
    # Process the project
    project_report = process_project(
        args.path, 
        dry_run=args.dry_run, 
        backup=args.backup,
        report=args.report,
        verbose=args.verbose
    )
    
    # Print summary
    print(f"\n{COLORS['BOLD']}Summary:{COLORS['RESET']}")
    print(f"Files analyzed: {project_report.files_analyzed}")
    print(f"Files with OpenAI calls: {project_report.files_with_openai_calls}")
    
    if not args.dry_run:
        print(f"Files modified: {project_report.files_modified}")
        print(f"Total calls logged: {project_report.total_calls_logged}")
    else:
        print(f"Total calls detected: {project_report.total_calls_detected}")
    
    if args.report:
        report_path = os.path.join(args.path, "openai_logger_report.md")
        print(f"\nDetailed report generated: {report_path}")

if __name__ == '__main__':
    main()

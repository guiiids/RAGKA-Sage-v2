"""Utility to inspect stored conversation history for a session."""

from services.session_memory import PostgresSessionMemory

memory_service = PostgresSessionMemory()

if __name__ == "__main__":
    session_id = "localtest"
    last_n = 10  # change as needed
    history = memory_service.get_history(session_id, last_n_turns=last_n)
    print(f"Conversation for session '{session_id}':")
    if not history:
        print("(No conversation turns found in Redis.)")
    else:
        for i, (user, assistant) in enumerate(history, 1):
            print(f"\nTurn {i}:")
            print(f"  User: {user}")
            print(f"  Assistant: {assistant}")

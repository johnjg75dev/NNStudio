"""
manage_users.py
CLI tool to manage NNStudio users (create, list, update, delete).
Usage:
    python manage_users.py list
    python manage_users.py create <username> <password>
    python manage_users.py update <username> <new_password>
    python manage_users.py delete <username>
"""
import sys
import argparse
from app import create_app, db
from app.models import User

def list_users():
    users = User.query.all()
    print(f"\n{'ID':<5} | {'Username':<20}")
    print("-" * 30)
    for u in users:
        print(f"{u.id:<5} | {u.username:<20}")
    print(f"\nTotal users: {len(users)}\n")

def create_user(username, password):
    if User.query.filter_by(username=username).first():
        print(f"Error: User '{username}' already exists.")
        return
    
    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    print(f"Success: User '{username}' created.")

def update_password(username, new_password):
    user = User.query.filter_by(username=username).first()
    if not user:
        print(f"Error: User '{username}' not found.")
        return
    
    user.set_password(new_password)
    db.session.commit()
    print(f"Success: Password updated for '{username}'.")

def delete_user(username):
    user = User.query.filter_by(username=username).first()
    if not user:
        print(f"Error: User '{username}' not found.")
        return
    
    db.session.delete(user)
    db.session.commit()
    print(f"Success: User '{username}' deleted.")

if __name__ == "__main__":
    app = create_app()
    
    parser = argparse.ArgumentParser(description="NNStudio User Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List
    subparsers.add_parser("list", help="List all users")

    # Create
    create_parser = subparsers.add_parser("create", help="Create a new user")
    create_parser.add_argument("username", help="Username")
    create_parser.add_argument("password", help="Password")

    # Update
    update_parser = subparsers.add_parser("update", help="Update user password")
    update_parser.add_argument("username", help="Username")
    update_parser.add_argument("password", help="New password")

    # Delete
    delete_parser = subparsers.add_parser("delete", help="Delete a user")
    delete_parser.add_argument("username", help="Username")

    args = parser.parse_parser = parser.parse_args()

    with app.app_context():
        if args.command == "list":
            list_users()
        elif args.command == "create":
            create_user(args.username, args.password)
        elif args.command == "update":
            update_password(args.username, args.password)
        elif args.command == "delete":
            delete_user(args.username)
        else:
            parser.print_help()

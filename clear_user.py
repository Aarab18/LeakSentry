from app import app, db, User

with app.app_context():
    try:
        # Get count of users before deletion
        user_count = User.query.count()
        print(f"Found {user_count} users in the database.")
        
        # Delete all users
        User.query.delete()
        db.session.commit()
        print("All users have been deleted successfully.")
        
        # Verify deletion
        user_count_after = User.query.count()
        print(f"Users after deletion: {user_count_after}")
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting users: {str(e)}")
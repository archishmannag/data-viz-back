"""
Supabase service for user management and storage.
Handles authentication, user management, and service usage tracking.
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from gotrue.errors import AuthError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class SupabaseService:
    """Service class for handling Supabase operations"""

    def __init__(self):
        """Initialize Supabase client"""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables"
            )

        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("Supabase client initialized successfully")

    # Authentication Methods
    async def sign_up(
        self, email: str, password: str, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Sign up a new user

        Args:
            email: User email
            password: User password
            metadata: Optional user metadata

        Returns:
            Dict containing user data and session info
        """
        try:
            response = self.client.auth.sign_up(
                {
                    "email": email,
                    "password": password,
                    "options": {"data": metadata} if metadata else {},
                }
            )

            if response.user:
                logger.info(f"User signed up successfully: {email}")
                return {
                    "success": True,
                    "user": response.user,
                    "session": response.session,
                    "message": "User created successfully",
                }
            else:
                return {"success": False, "message": "Failed to create user"}

        except AuthError as e:
            logger.error(f"Sign up error: {e}")
            return {"success": False, "message": str(e)}

    async def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """
        Sign in an existing user

        Args:
            email: User email
            password: User password

        Returns:
            Dict containing user data and session info
        """
        try:
            response = self.client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )

            if response.user:
                logger.info(f"User signed in successfully: {email}")
                result = {
                    "success": True,
                    "user": response.user,
                    "session": response.session,
                }

                # Add tokens if session exists
                if response.session:
                    result["access_token"] = response.session.access_token
                    result["refresh_token"] = response.session.refresh_token

                return result
            else:
                return {"success": False, "message": "Invalid credentials"}

        except AuthError as e:
            logger.error(f"Sign in error: {e}")
            return {"success": False, "message": str(e)}

    async def sign_out(self) -> Dict[str, Any]:
        """Sign out current user"""
        try:
            self.client.auth.sign_out()
            logger.info("User signed out successfully")
            return {"success": True, "message": "User signed out successfully"}
        except AuthError as e:
            logger.error(f"Sign out error: {e}")
            return {"success": False, "message": str(e)}

    async def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user"""
        try:
            user = self.client.auth.get_user()
            if user and user.user:
                return {"success": True, "user": user.user}
            return None
        except AuthError as e:
            logger.error(f"Get user error: {e}")
            return None

    async def refresh_session(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh user session"""
        try:
            response = self.client.auth.refresh_session(refresh_token)
            result = {
                "success": True,
                "session": response.session,
            }

            # Add tokens if session exists
            if response.session:
                result["access_token"] = response.session.access_token
                result["refresh_token"] = response.session.refresh_token

            return result
        except AuthError as e:
            logger.error(f"Refresh session error: {e}")
            return {"success": False, "message": str(e)}

    # User Management Methods
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile from database"""
        try:
            response = (
                self.client.table("user_profiles")
                .select("*")
                .eq("user_id", user_id)
                .single()
                .execute()
            )
            if response.data:
                return response.data
            return None
        except Exception as e:
            logger.error(f"Get user profile error: {e}")
            return None

    async def create_user_profile(
        self, user_id: str, profile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create user profile in database"""
        try:
            profile_data["user_id"] = user_id
            profile_data["created_at"] = datetime.now(timezone.utc).isoformat()

            response = self.client.table("user_profiles").insert(profile_data).execute()

            if response.data:
                logger.info(f"User profile created for user: {user_id}")
                return {"success": True, "profile": response.data[0]}
            return {"success": False, "message": "Failed to create user profile"}
        except Exception as e:
            logger.error(f"Create user profile error: {e}")
            return {"success": False, "message": str(e)}

    async def update_user_profile(
        self, user_id: str, profile_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update user profile in database"""
        try:
            profile_data["updated_at"] = datetime.now(timezone.utc).isoformat()

            response = (
                self.client.table("user_profiles")
                .update(profile_data)
                .eq("user_id", user_id)
                .execute()
            )

            if response.data:
                logger.info(f"User profile updated for user: {user_id}")
                return {"success": True, "profile": response.data[0]}
            return {"success": False, "message": "Failed to update user profile"}
        except Exception as e:
            logger.error(f"Update user profile error: {e}")
            return {"success": False, "message": str(e)}

    # Service Usage Tracking Methods
    async def create_service_record(
        self,
        user_id: str,
        file_names: List[str],
        ai_response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a service usage record

        Args:
            user_id: User ID
            file_names: List of uploaded file names
            ai_response: AI response data
            metadata: Optional metadata (file sizes, processing time, etc.)

        Returns:
            Dict containing the created record
        """
        try:
            record_data = {
                "user_id": user_id,
                "file_names": file_names,
                "ai_response": ai_response,
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            response = (
                self.client.table("service_usage_records").insert(record_data).execute()
            )

            if response.data:
                logger.info(f"Service record created for user: {user_id}")
                return {"success": True, "record": response.data[0]}
            return {"success": False, "message": "Failed to create service record"}
        except Exception as e:
            logger.error(f"Create service record error: {e}")
            return {"success": False, "message": str(e)}

    async def get_user_service_records(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get user's service usage records"""
        try:
            response = (
                self.client.table("service_usage_records")
                .select("file_names, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .offset(offset)
                .execute()
            )

            return response.data or []
        except Exception as e:
            logger.error(f"Get user service records error: {e}")
            return []

    async def get_service_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific service record"""
        try:
            response = (
                self.client.table("service_usage_records")
                .select("*")
                .eq("id", record_id)
                .single()
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Get service record error: {e}")
            return None

    async def update_service_record(
        self, record_id: str, update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a service record"""
        try:
            update_data["updated_at"] = datetime.now(timezone.utc).isoformat()

            response = (
                self.client.table("service_usage_records")
                .update(update_data)
                .eq("id", record_id)
                .execute()
            )

            if response.data:
                return {"success": True, "record": response.data[0]}
            return {"success": False, "message": "Failed to update service record"}
        except Exception as e:
            logger.error(f"Update service record error: {e}")
            return {"success": False, "message": str(e)}

    async def delete_service_record(
        self, record_id: str, user_id: str
    ) -> Dict[str, Any]:
        """Delete a service record (only by the owner)"""
        try:
            response = (
                self.client.table("service_usage_records")
                .delete()
                .eq("id", record_id)
                .eq("user_id", user_id)
                .execute()
            )

            if response.data:
                return {
                    "success": True,
                    "message": "Service record deleted successfully",
                }
            return {
                "success": False,
                "message": "Failed to delete service record or record not found",
            }
        except Exception as e:
            logger.error(f"Delete service record error: {e}")
            return {"success": False, "message": str(e)}

    # Analytics and Statistics Methods
    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user usage statistics"""
        try:
            # Get total records count
            total_records = (
                self.client.table("service_usage_records")
                .select("*")
                .eq("user_id", user_id)
                .execute()
            )

            # Get records from last 30 days
            thirty_days_ago = datetime.now(timezone.utc).replace(day=1).isoformat()
            recent_records = (
                self.client.table("service_usage_records")
                .select("*")
                .eq("user_id", user_id)
                .gte("created_at", thirty_days_ago)
                .execute()
            )

            # Calculate statistics
            total_files_processed = 0
            if recent_records.data:
                for record in recent_records.data:
                    total_files_processed += len(record.get("file_names", []))

            return {
                "total_records": len(total_records.data) if total_records.data else 0,
                "records_last_30_days": len(recent_records.data)
                if recent_records.data
                else 0,
                "total_files_processed_last_30_days": total_files_processed,
                "last_activity": recent_records.data[0]["created_at"]
                if recent_records.data
                else None,
            }
        except Exception as e:
            logger.error(f"Get user statistics error: {e}")
            return {
                "total_records": 0,
                "records_last_30_days": 0,
                "total_files_processed_last_30_days": 0,
                "last_activity": None,
            }

    # Database Helper Methods
    async def initialize_database(self) -> Dict[str, Any]:
        """Initialize database tables if they don't exist"""
        try:
            # This would typically be handled by Supabase migrations
            # But we can check if tables exist and provide helpful error messages

            # Test user_profiles table
            try:
                self.client.table("user_profiles").select("id").limit(1).execute()
            except Exception:
                logger.warning("user_profiles table might not exist")

            # Test service_usage_records table
            try:
                self.client.table("service_usage_records").select("id").limit(
                    1
                ).execute()
            except Exception:
                logger.warning("service_usage_records table might not exist")

            return {"success": True, "message": "Database check completed"}
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            return {"success": False, "message": str(e)}


# Global service instance
supabase_service = SupabaseService()


# Convenience functions for easy access
async def authenticate_user(email: str, password: str) -> Dict[str, Any]:
    """Convenience function to authenticate user"""
    return await supabase_service.sign_in(email, password)


async def create_user(
    email: str, password: str, metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """Convenience function to create user"""
    return await supabase_service.sign_up(email, password, metadata)


async def log_service_usage(
    user_id: str,
    file_names: List[str],
    ai_response: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to log service usage"""
    return await supabase_service.create_service_record(
        user_id, file_names, ai_response, metadata
    )


async def get_user_history(
    user_id: str, limit: int = 50, offset: int = 0
) -> List[Dict[str, Any]]:
    """Convenience function to get user history"""
    return await supabase_service.get_user_service_records(user_id, limit, offset)


if __name__ == "__main__":
    # Test the service
    async def test_service():
        try:
            # Initialize database
            result = await supabase_service.initialize_database()
            print("Database initialization:", result)

            # Test authentication (you would use real credentials)
            # auth_result = await authenticate_user("test@example.com", "password123")
            # print("Authentication test:", auth_result)

        except Exception as e:
            print(f"Test error: {e}")

    # Run test
    asyncio.run(test_service())

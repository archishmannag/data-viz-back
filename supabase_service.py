"""
Supabase service for user management and storage.
Handles authentication, user management, and service usage tracking.
"""

import os
import asyncio
import uuid
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
                .select("id, file_names, created_at")
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

    async def get_service_record_by_share_id(
        self, share_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a service record by its unique share_id

        Args:
            share_id: The unique share ID of the record

        Returns:
            Service record data or None if not found
        """
        try:
            response = (
                self.client.table("service_usage_records")
                .select("*")
                .eq("share_id", share_id)
                .single()
                .execute()
            )

            if response.data:
                logger.info(f"Service record found for share_id: {share_id}")
                return response.data
            else:
                logger.info(f"No service record found for share_id: {share_id}")
                return None

        except Exception as e:
            logger.error(f"Error getting service record by share_id {share_id}: {e}")
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

    # File Storage Methods
    async def upload_file(
        self,
        user_id: str,
        file_path: str,
        file_content: bytes,
        bucket_name: str = "user-files",
    ) -> Dict[str, Any]:
        """
        Upload a file to Supabase storage

        Args:
            user_id: User ID
            file_path: Original file path/name
            file_content: File content as bytes
            bucket_name: Storage bucket name

        Returns:
            Dict containing upload result and file info
        """
        try:
            # Generate unique filename with user ID prefix and timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            storage_filename = (
                f"{user_id}/{timestamp}_{unique_id}_{os.path.basename(file_path)}"
            )

            # Upload file to storage
            response = self.client.storage.from_(bucket_name).upload(
                storage_filename, file_content
            )

            if response:
                # Get public URL for the file
                public_url = self.client.storage.from_(bucket_name).get_public_url(
                    storage_filename
                )

                logger.info(f"File uploaded successfully: {storage_filename}")
                return {
                    "success": True,
                    "storage_path": storage_filename,
                    "public_url": public_url,
                    "original_filename": os.path.basename(file_path),
                    "file_size": len(file_content),
                    "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "File uploaded successfully",
                }
            else:
                return {"success": False, "message": "Failed to upload file"}
        except Exception as e:
            logger.error(f"File upload error: {e}")
            return {"success": False, "message": str(e)}

    async def upload_multiple_files(
        self,
        user_id: str,
        files_data: List[Dict[str, Any]],
        bucket_name: str = "user-files",
    ) -> List[Dict[str, Any]]:
        """
        Upload multiple files concurrently

        Args:
            user_id: User ID
            files_data: List of dicts with 'path' and 'content' keys
            bucket_name: Storage bucket name

        Returns:
            List of upload results
        """
        upload_tasks = []

        for file_data in files_data:
            task = self.upload_file(
                user_id=user_id,
                file_path=file_data["path"],
                file_content=file_data["content"],
                bucket_name=bucket_name,
            )
            upload_tasks.append(task)

        # Execute all uploads concurrently
        results = await asyncio.gather(*upload_tasks, return_exceptions=True)

        # Handle any exceptions that occurred during upload
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"File upload failed for {files_data[i]['path']}: {result}"
                )
                processed_results.append(
                    {
                        "success": False,
                        "original_filename": os.path.basename(files_data[i]["path"]),
                        "message": str(result),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    async def delete_file(
        self, storage_path: str, bucket_name: str = "user-files"
    ) -> Dict[str, Any]:
        """
        Delete a file from Supabase storage

        Args:
            storage_path: Path of the file in storage
            bucket_name: Storage bucket name

        Returns:
            Dict containing deletion result
        """
        try:
            response = self.client.storage.from_(bucket_name).remove([storage_path])

            if response:
                logger.info(f"File deleted successfully: {storage_path}")
                return {"success": True, "message": "File deleted successfully"}
            else:
                return {"success": False, "message": "Failed to delete file"}
        except Exception as e:
            logger.error(f"File deletion error: {e}")
            return {"success": False, "message": str(e)}

    async def download_file(
        self, storage_path: str, bucket_name: str = "user-files"
    ) -> Dict[str, Any]:
        """
        Download a file from Supabase storage

        Args:
            storage_path: Path of the file in storage
            bucket_name: Storage bucket name

        Returns:
            Dict containing file content and metadata
        """
        try:
            response = self.client.storage.from_(bucket_name).download(storage_path)

            if response:
                logger.info(f"File downloaded successfully: {storage_path}")
                return {
                    "success": True,
                    "content": response,
                    "storage_path": storage_path,
                    "message": "File downloaded successfully",
                }
            else:
                return {"success": False, "message": "Failed to download file"}

        except Exception as e:
            logger.error(f"File download error: {e}")
            return {"success": False, "message": str(e)}

    async def get_user_files(
        self, user_id: str, bucket_name: str = "user-files", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get list of files uploaded by a user

        Args:
            user_id: User ID
            bucket_name: Storage bucket name
            limit: Maximum number of files to return

        Returns:
            List of file information
        """
        try:
            # List files in user's directory
            response = self.client.storage.from_(bucket_name).list(
                f"{user_id}/",
                {"limit": limit, "sortBy": {"column": "created_at", "order": "desc"}},
            )

            if response:
                files = []
                for file_info in response:
                    # Skip if empty placeholder file from supa
                    if file_info["name"] == ".emptyFolderPlaceholder":
                        continue
                    # Safely handle metadata field
                    metadata = file_info.get("metadata", {})
                    file_size = 0
                    if isinstance(metadata, dict):
                        file_size = metadata.get("size", 0)

                    # Extract original filename from storage filename
                    storage_name = file_info["name"]
                    # Storage format: timestamp_uuid_originalname.ext
                    parts = storage_name.split("_", 3)
                    original_name = parts[3] if len(parts) >= 4 else storage_name

                    files.append(
                        {
                            "name": file_info["name"],
                            "original_name": original_name,
                            "storage_path": f"{user_id}/{file_info['name']}",
                            "size": file_size,
                            "created_at": file_info.get("created_at"),
                            "updated_at": file_info.get("updated_at"),
                            "public_url": self.client.storage.from_(
                                bucket_name
                            ).get_public_url(f"{user_id}/{file_info['name']}"),
                        }
                    )
                return files
            else:
                return []

        except Exception as e:
            logger.error(f"Get user files error: {e}")
            return []

    async def get_user_files_with_content(
        self, user_id: str, file_names: List[str], bucket_name: str = "user-files"
    ) -> List[Dict[str, Any]]:
        """
        Get specific files by their storage names with content

        Args:
            user_id: User ID
            file_names: List of storage file names to retrieve
            bucket_name: Storage bucket name

        Returns:
            List of file data with content
        """
        try:
            download_tasks = []

            for file_name in file_names:
                storage_path = f"{user_id}/{file_name}"
                task = self.download_file(storage_path, bucket_name)
                download_tasks.append(task)

            # Execute all downloads concurrently
            results = await asyncio.gather(*download_tasks, return_exceptions=True)

            # Process results
            files_data = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to download {file_names[i]}: {result}")
                    files_data.append(
                        {
                            "success": False,
                            "storage_name": file_names[i],
                            "message": str(result),
                        }
                    )
                elif isinstance(result, dict) and result.get("success"):
                    files_data.append(
                        {
                            "success": True,
                            "storage_name": file_names[i],
                            "content": result["content"],
                            "storage_path": result["storage_path"],
                        }
                    )
                else:
                    message = "Unknown error"
                    if isinstance(result, dict):
                        message = result.get("message", "Unknown error")
                    files_data.append(
                        {
                            "success": False,
                            "storage_name": file_names[i],
                            "message": message,
                        }
                    )

            return files_data

        except Exception as e:
            logger.error(f"Get user files with content error: {e}")
            return [
                {"success": False, "storage_name": name, "message": str(e)}
                for name in file_names
            ]


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


async def upload_user_files(
    user_id: str, files_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Convenience function to upload multiple files for a user"""
    return await supabase_service.upload_multiple_files(user_id, files_data)


async def upload_user_file(
    user_id: str, file_path: str, file_content: bytes
) -> Dict[str, Any]:
    """Convenience function to upload a single file for a user"""
    return await supabase_service.upload_file(user_id, file_path, file_content)


async def get_user_uploaded_files(user_id: str) -> List[Dict[str, Any]]:
    """Convenience function to get user's uploaded files"""
    return await supabase_service.get_user_files(user_id)


async def get_visualization_by_share_id(share_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get visualization by share_id"""
    return await supabase_service.get_service_record_by_share_id(share_id)


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

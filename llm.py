from fastapi import APIRouter, UploadFile, Depends, Form
from google import genai
from google.genai.types import Content, Part, GenerateContentConfig
from typing import Dict, Any, List, Tuple
import fitz  # PyMuPDF
import re
import os
import json
import pandas as pd
import time
import math
import asyncio
import logging

import chunker
from supa import get_current_user_optional
from supabase_service import supabase_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Create router
processing_router = APIRouter(prefix="", tags=["document-processing"])
system_router = APIRouter(prefix="/system", tags=["system"])
visualization_router = APIRouter(prefix="/visualization", tags=["visualization"])


# Exception types for different error scenarios
class QuotaExceededException(Exception):
    """Raised when API quota is exceeded"""

    pass


class RetryableException(Exception):
    """Raised for errors that can be retried"""

    pass


def is_quota_exceeded_error(error: Exception) -> bool:
    """
    Check if the error indicates quota exceeded or billing issues

    Args:
        error: The exception to check

    Returns:
        bool: True if error indicates quota/billing issues
    """
    error_str = str(error).lower()
    quota_indicators = [
        "quota exceeded",
        "insufficient quota",
        "billing",
        "quota_exceeded",
        "resource_exhausted",
        "too many requests",
    ]
    return any(indicator in error_str for indicator in quota_indicators)


def is_retryable_error(error: Exception) -> bool:
    """
    Check if the error is retryable (temporary issues)

    Args:
        error: The exception to check

    Returns:
        bool: True if error can be retried
    """
    if is_quota_exceeded_error(error):
        return False

    error_str = str(error).lower()
    retryable_indicators = [
        "timeout",
        "connection",
        "network",
        "unavailable",
        "internal error",
        "server error",
        "503",
        "502",
        "500",
        "overloaded",
        "rate limit",
        "temporarily unavailable",
    ]
    return any(indicator in error_str for indicator in retryable_indicators)


async def cleanup_uploaded_files(upload_results: List[Dict[str, Any]]) -> None:
    """
    Clean up uploaded files from storage when quota is exceeded

    Args:
        upload_results: List of upload results containing storage paths
    """
    if not upload_results:
        return

    logger.info("Cleaning up uploaded files due to quota exceeded error...")

    for result in upload_results:
        if result.get("success") and result.get("storage_path"):
            try:
                storage_path = result["storage_path"]
                delete_result = await supabase_service.delete_file(storage_path)
                if delete_result.get("success"):
                    logger.info(f"Successfully deleted file: {storage_path}")
                else:
                    logger.error(
                        f"Failed to delete file {storage_path}: {delete_result.get('message')}"
                    )
            except Exception as e:
                logger.error(
                    f"Error deleting file {result.get('storage_path')}: {str(e)}"
                )


async def generate_chart_response_with_retry(
    content: List[str], num_files: int, max_retries: int = 3, base_delay: float = 1.0
) -> Tuple[Dict[str, Any], bool]:
    """
    Generate chart response with retry logic and exponential backoff

    Args:
        content: List of content strings
        num_files: Number of files
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds

    Returns:
        Tuple of (response_dict, is_quota_exceeded)
    """
    last_exception = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            logger.info(
                f"AI response generation attempt {attempt + 1}/{max_retries + 1}"
            )
            response = generate_chart_response(content, num_files)
            logger.info("AI response generated successfully")
            return response, False

        except Exception as e:
            last_exception = e
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")

            # Check if quota exceeded
            if is_quota_exceeded_error(e):
                logger.error("Quota exceeded - cannot retry")
                return {
                    "error": "quota_exceeded",
                    "message": "API quota exceeded. Please try again later or upgrade your plan.",
                    "details": str(e),
                }, True

            # Check if retryable
            if not is_retryable_error(e):
                logger.error(f"Non-retryable error: {str(e)}")
                return {
                    "error": "non_retryable",
                    "message": "An error occurred that cannot be retried.",
                    "details": str(e),
                }, False

            # If this is the last attempt, don't wait
            if attempt == max_retries:
                break

            # Exponential backoff with jitter
            delay = base_delay * (2**attempt) + (0.1 * attempt)  # Add small jitter
            logger.info(f"Retrying in {delay:.1f} seconds...")
            await asyncio.sleep(delay)

    # All attempts failed
    logger.error(
        f"All {max_retries + 1} attempts failed. Last error: {str(last_exception)}"
    )
    return {
        "error": "max_retries_exceeded",
        "message": f"Failed to generate response after {max_retries + 1} attempts.",
        "details": str(last_exception) if last_exception else "Unknown error",
    }, False


async def upload_files_to_storage(
    user_id: str, files_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Upload files to Supabase storage asynchronously

    Args:
        user_id: User ID
        files_data: List of file data with 'path' and 'content' keys

    Returns:
        List of upload results
    """
    if not files_data:
        return []

    try:
        upload_results = await supabase_service.upload_multiple_files(
            user_id=user_id, files_data=files_data
        )
        return upload_results
    except Exception as e:
        logger.error(f"Error uploading files to storage: {str(e)}")
        return [
            {
                "success": False,
                "message": str(e),
                "original_filename": file_data.get("path", "unknown"),
            }
            for file_data in files_data
        ]


def split_string_preserve_words(text: str, n: int) -> list[str]:
    words = text.split()
    avg_words = math.ceil(len(words) / n)

    chunks = []
    for i in range(0, len(words), avg_words):
        chunk = " ".join(words[i : i + avg_words])
        chunks.append(chunk)

    # Ensure exactly n chunks (pad last ones if needed)
    while len(chunks) < n:
        chunks.append("")

    return chunks


def generate_chart_response(content: List[str], num_files: int) -> Dict[str, Any]:
    combined_content = "\n\n".join(content)
    prompt_text = (
        """
Retrieved document chunks:
"""
        + combined_content
        + """
User query:
    Using the above document chunks, create as many charts as possible.
    Return the charts in JSON format with the following structure:
{
  "dashboard": {
    "title": "Dashboard Title",
    "description": "Brief description of what this dashboard shows",
    "kpis": [
      {
        "id": "kpi1",
        "title": "Total Revenue",
        "value": 1250000,
        "unit": "$",
        "change": "+15.2%",
        "changeType": "positive|negative|neutral",
        "icon": "dollar-sign",
        "color": "#10B981"
      }
    ],
    "charts": [
      {
        "id": "chart1",
        "type": "AreaChart|BarChart|LineChart|ComposedChart|PieChart|RadarChart|RadialBarChart|ScatterChart|FunnelChart|SankeyChart",
        "title": "Chart Title",
        "size": "full|half|third|quarter",
        "chartConfig": {
          "xAxis": {
            "dataKey": "fieldName",
            "label": "X Axis Label",
            "type": "category|number|time"
          },
          "yAxis": {
            "label": "Y Axis Label",
            "domain": ["auto", "auto"] // or specific range
          },
          "series": [
            {
              "dataKey": "field1",
              "name": "Series Name",
              "type": "bar|line|area",
              "color": "#3B82F6",
              "fill": "#3B82F6",
              "stroke": "#3B82F6"
            }
          ],
          "composedComponents": [
            {
              "type": "Bar|Line|Area",
              "dataKey": "field1",
              "name": "Component Name",
              "color": "#3B82F6",
              "fill": "#3B82F6",
              "stroke": "#3B82F6"
            }
          ] // Only required for ComposedChart type
        },
        "data": [
          {"category": "Jan", "value1": 100, "value2": 150, "color": "#3B82F6"}, // Color for pie charts and funnelcharts
          {"category": "Feb", "value1": 120, "value2": 180, "color": "#3B82F6"}
        ]
      }
    ],
    "tables": [
      {
        "id": "table1",
        "title": "Detailed Data",
        "columns": [
          {"key": "name", "label": "Name", "type": "text"},
          {"key": "value", "label": "Value", "type": "number", "format": "currency"}
        ],
        "data": [],
        "pagination": true,
        "sortable": true
      }
    ],
    "optimizationSuggestions": [
      {
        "id": "suggestion1",
        "title": "Cost Reduction Opportunity",
        "category": "cost|efficiency|performance|risk|quality",
        "impact": "high|medium|low",
        "savings": {
          "value": 50000,
          "unit": "$",
          "percentage": "15%",
          "timeframe": "annually"
        },
        "description": "Detailed explanation of the optimization opportunity",
        "implementation": "Step-by-step guide on how to implement this optimization",
        "metrics": [
          "Current metric: 82% efficiency",
          "Target metric: 95% efficiency",
          "Expected timeline: 3-6 months"
        ],
        "priority": "high|medium|low",
        "confidence": "high|medium|low",
        "tags": ["drilling", "efficiency", "cost-optimization"],
        "actionable": true,
        "color": "#10B981"
      }
    ],
    "insights": {
      "summary": "Overall dashboard insights and key takeaways",
      "trends": [
        "Trend 1: Increasing efficiency over time",
        "Trend 2: Cost reduction opportunities identified"
      ],
      "alerts": [
        {
          "type": "warning|error|info|success",
          "message": "Alert message",
          "severity": "high|medium|low",
          "action": "Recommended action to take"
        }
      ],
      "recommendations": [
        "Recommendation 1: Focus on drilling optimization",
        "Recommendation 2: Implement predictive maintenance"
      ]
    }
  }
}

The data for Sankey must be in the following format in accordance to Recharts:
{
    'nodes': [
        { 'name': 'Visit' },
        { 'name': 'Direct-Favourite' },
        { 'name': 'Page-Click' },
        { 'name': 'Detail-Favourite' },
        { 'name': 'Lost' },
    ],
    'links': [
        { 'source': 0, 'target': 1, 'value': 3728.3 },
        { 'source': 0, 'target': 2, 'value': 354170 },
        { 'source': 2, 'target': 3, 'value': 62429 },
        { 'source': 2, 'target': 4, 'value': 291741 },
    ],
}
"""
    )
    count = client.models.count_tokens(
        model="gemini-2.0-flash", contents=[prompt_text]
    ).total_tokens
    if count and count > 1_000_000:
        num = count // 1_000_000
        chunks = split_string_preserve_words(combined_content, num)
        prompt_parts = [
            Content(
                role="user",
                parts=[Part(text=chunk)],
            )
            for chunk in chunks
        ]
        prompt_parts.insert(
            0,
            Content(
                role="user",
                parts=[
                    Part(
                        text="I will send the entire prompt in chunks, please wait until I finish sending all the chunks."
                    )
                ],
            ),
        )
    else:
        prompt_parts = [
            Content(
                role="user",
                parts=[Part(text=prompt_text)],
            )
        ]

    json_pattern = r"(\{.*\})"
    config = GenerateContentConfig(
        temperature=0.2,
        system_instruction="""
You are an expert assistant using a retrieval-augmented generation system. Use the following retrieved document chunks to answer the user's query.
Context: You are analyzing data to create comprehensive dashboards similar to business intelligence tools.
Instructions:
1. Analyze the data structure and identify key metrics, trends, and relationships
2. Create a variety of visualizations including:
   - KPI cards for key metrics
   - Time series charts for trends
   - Comparison charts (bar/column) for categorical data
   - Distribution charts (pie/donut) for proportions
   - Tables for detailed data views
   - Categorical breakdowns if applicable

3. For each visualization, consider:
   - What story does this data tell?
   - What insights can be derived?
   - How does it relate to business objectives?

4. Return response in the enhanced JSON structure.

5. We will need detailed results. So generate as much as you can, do not be short. Try to be as expressive as possible.

6. The JSON objects like charts, tables, KPIs, optimizationSuggestions, etc. should be complete and valid JSON objects. Do not return any incomplete JSON objects. They are all optional though. If you find that the data provided does not require a chart or table, you can skip that part.
""",
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt_parts, config=config
    )
    result: Dict[str, Any] = {}
    if response.text:
        json_match = re.search(json_pattern, response.text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            result = json.loads(json_str)

    return result


def process_pdf(file_path: str) -> List[str]:
    # Open document with PyMuPDF
    doc = fitz.open(file_path)

    # First try direct extraction
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()  # type: ignore - PyMuPDF method
        pages.append(text)
    text = ""
    for raw in pages:
        text += raw + "\n\n"

    # Apply chunking
    chunks = chunker.chunk_text(text, 10000)
    return chunks


def process_excel(file_path: str) -> List[str]:
    """Process Excel files with better chunking by rows"""
    try:
        # Read Excel file
        df_dict = pd.read_excel(file_path, sheet_name=None)
        filename = os.path.basename(file_path)
        chunks: List[str] = []

        for sheet_name, df in df_dict.items():
            total_rows = len(df)

            # Add sheet overview with column information
            overview_text = f"Sheet '{sheet_name}' overview from file '{filename}':\n"
            overview_text += f"Total rows: {total_rows}\n"
            overview_text += f"Columns: {', '.join(df.columns)}\n\n"

            # Add column statistics
            if len(df.columns) > 0:
                overview_text += "Column statistics:\n"
                for col in df.columns:
                    overview_text += f"- {col}: "
                    try:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            overview_text += (
                                f"numeric, range [{df[col].min()}-{df[col].max()}], "
                            )
                            overview_text += f"mean: {df[col].mean():.2f}\n"
                        else:
                            unique_vals = df[col].nunique()
                            overview_text += (
                                f"text/categorical, {unique_vals} unique values\n"
                            )
                    except Exception:
                        overview_text += "unknown type\n"

            # Add this overview as a chunk
            chunks.append(overview_text)

            rows_text: List[str] = []
            rows_text.append(
                "| Row | " + " | ".join(str(col) for col in df.columns) + " |"
            )
            for idx, (i, row) in enumerate(df.iterrows(), start=1):
                row_text = (
                    f"| {idx} | " + " | ".join(str(val) for val in row.values) + " |"
                )
                rows_text.append(row_text)
            chunks.append("\n".join(rows_text))

        return chunks
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        # Return empty list on error instead of trying text processing
        return []


@processing_router.post("/generate")
async def generate_response(
    files: List[UploadFile] = [],
    existing_files: str = Form("[]"),  # JSON string of existing file names
    current_user=Depends(get_current_user_optional),
):
    """Generate a response based on uploaded files and/or existing files.

    Args:
        files (List[UploadFile]): New files to process and upload.
        existing_files (str): JSON string array of storage names of existing files to reuse.
    """
    start_time = time.time()

    # Parse existing files from JSON string
    try:
        existing_file_list = json.loads(existing_files) if existing_files else []
        if not isinstance(existing_file_list, list):
            existing_file_list = []
    except json.JSONDecodeError:
        existing_file_list = []

    file_locations = []
    file_names = []
    file_sizes = []
    files_content = []  # Store file content for storage upload
    file_name_mapping = {}  # Map original names to storage names

    # Process new uploaded files
    for i, file in enumerate(files):
        filename = file.filename or f"uploaded_file_{i}"
        file_location = f"/tmp/{filename}"
        file_locations.append(file_location)
        file_names.append(filename)

    # Save new files and track sizes
    for i, file_location in enumerate(file_locations):
        content = await files[i].read()
        file_sizes.append(len(content))
        files_content.append({"path": file_location, "content": content})
        with open(file_location, "wb") as f:
            f.write(content)

    # Process existing files from storage (if user is authenticated)
    existing_file_data = []
    if current_user and existing_file_list:
        try:
            existing_file_data = await supabase_service.get_user_files_with_content(
                current_user.id, existing_file_list
            )

            # Save existing files to temp locations for processing
            for file_data in existing_file_data:
                if file_data.get("success") and file_data.get("content"):
                    storage_name = file_data["storage_name"]
                    # Extract original name from storage name (timestamp_uuid_originalname.ext)
                    parts = storage_name.split("_", 3)
                    original_name = parts[3] if len(parts) >= 4 else storage_name

                    temp_location = f"/tmp/existing_{original_name}"
                    file_locations.append(temp_location)
                    file_names.append(original_name)
                    file_sizes.append(len(file_data["content"]))

                    # Save to temp file for processing
                    with open(temp_location, "wb") as f:
                        f.write(file_data["content"])

                    # Track the mapping
                    file_name_mapping[original_name] = storage_name
                else:
                    logger.warning(
                        f"Failed to retrieve existing file: {file_data.get('storage_name', 'unknown')}"
                    )
        except Exception as e:
            logger.error(f"Error retrieving existing files: {str(e)}")

    # Process files for text extraction
    text_chunks: List[str] = []
    for file_location in file_locations:
        text_chunks.append(f"File: {file_location}\n")
        if file_location.endswith(".pdf"):
            text_chunks.extend(process_pdf(file_location))
        elif file_location.endswith(".xlsx") or file_location.endswith(".xls"):
            text_chunks.extend(process_excel(file_location))

    # Create tasks for parallel execution
    tasks = []

    # Task 1: Generate AI response with retry logic
    ai_response_task = asyncio.create_task(
        generate_chart_response_with_retry(text_chunks, len(file_locations))
    )
    tasks.append(ai_response_task)

    # Task 2: Upload NEW files to storage (only if user is authenticated and there are new files)
    upload_task = None
    if current_user and files_content:  # Only upload new files, not existing ones
        upload_task = asyncio.create_task(
            upload_files_to_storage(current_user.id, files_content)
        )
        tasks.append(upload_task)

    # Execute tasks in parallel
    if upload_task:
        # Wait for both AI response and file upload
        results = await asyncio.gather(
            ai_response_task, upload_task, return_exceptions=True
        )
        ai_response_result, upload_results = results

        # Handle upload results (log any errors but don't fail the request)
        if isinstance(upload_results, Exception):
            logger.error(f"File upload failed: {upload_results}")
            upload_results = []
        elif upload_results and isinstance(upload_results, list):
            successful_uploads = [r for r in upload_results if r.get("success", False)]
            failed_uploads = [r for r in upload_results if not r.get("success", False)]

            if successful_uploads:
                logger.info(
                    f"Successfully uploaded {len(successful_uploads)} files to storage"
                )
            if failed_uploads:
                logger.warning(
                    f"Failed to upload {len(failed_uploads)} files to storage"
                )
        else:
            upload_results = []
    else:
        # Only wait for AI response if no user is logged in
        ai_response_result = await ai_response_task
        upload_results = []

    # Handle AI response result
    if isinstance(ai_response_result, Exception):
        logger.error(f"AI response generation failed: {ai_response_result}")
        ai_response = {"error": "Failed to generate response"}
        is_quota_exceeded = False
    elif isinstance(ai_response_result, tuple) and len(ai_response_result) == 2:
        ai_response, is_quota_exceeded = ai_response_result
    else:
        # Fallback for unexpected result format
        ai_response = (
            ai_response_result if ai_response_result else {"error": "Unknown error"}
        )
        is_quota_exceeded = False

    # Handle quota exceeded case - cleanup uploaded files
    if is_quota_exceeded and upload_results:
        logger.warning("Quota exceeded - cleaning up uploaded files")
        await cleanup_uploaded_files(upload_results)
        # Clear upload results since files were deleted
        upload_results = []

        # Return quota error immediately
        from fastapi import HTTPException

        raise HTTPException(
            status_code=429,
            detail={
                "error": "quota_exceeded",
                "message": "API quota exceeded. Please try again later or upgrade your plan.",
                "uploaded_files_cleaned": True,
            },
        )

    # Handle other AI response errors
    if isinstance(ai_response, dict) and ai_response.get("error"):
        error_type = ai_response.get("error")
        if error_type in ["max_retries_exceeded", "non_retryable"]:
            # For serious errors, we should also clean up uploaded files
            if upload_results:
                logger.warning(
                    f"AI response failed with {error_type} - cleaning up uploaded files"
                )
                await cleanup_uploaded_files(upload_results)
                upload_results = []

            from fastapi import HTTPException

            status_code = 503 if error_type == "max_retries_exceeded" else 500
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": error_type,
                    "message": ai_response.get(
                        "message", "AI response generation failed"
                    ),
                    "details": ai_response.get("details"),
                    "uploaded_files_cleaned": bool(upload_results),
                },
            )

    # Calculate processing time
    processing_time = time.time() - start_time
    record = None
    # Track service usage if user is authenticated and response is successful
    if current_user and isinstance(ai_response, dict) and not ai_response.get("error"):
        # Calculate upload statistics safely
        files_uploaded = 0
        upload_errors = 0
        if upload_results and isinstance(upload_results, list):
            files_uploaded = len([r for r in upload_results if r.get("success", False)])
            upload_errors = len(
                [r for r in upload_results if not r.get("success", False)]
            )

        # Create comprehensive file mapping for metadata
        all_file_info = []

        # Add new uploaded files
        for i, filename in enumerate(file_names[: len(files)]):  # Only new files
            file_info = {
                "original_name": filename,
                "type": "new_upload",
                "size": file_sizes[i] if i < len(file_sizes) else 0,
                "file_type": filename.split(".")[-1] if "." in filename else "unknown",
            }
            # Add storage name if upload was successful
            if (
                upload_results
                and i < len(upload_results)
                and upload_results[i].get("success")
            ):
                file_info["storage_name"] = (
                    upload_results[i].get("storage_path", "").split("/")[-1]
                )
                file_info["storage_path"] = upload_results[i].get("storage_path", "")
            all_file_info.append(file_info)

        # Add existing reused files
        for filename in file_names[len(files) :]:  # Only existing files
            file_info = {
                "original_name": filename,
                "storage_name": file_name_mapping.get(filename, ""),
                "type": "reused_file",
                "file_type": filename.split(".")[-1] if "." in filename else "unknown",
            }
            all_file_info.append(file_info)

        metadata = {
            "processing_time_seconds": processing_time,
            "file_sizes_bytes": file_sizes,
            "total_files": len(file_locations),
            "new_files_count": len(files),
            "reused_files_count": len(existing_file_list),
            "endpoint": "/processing/generate",
            "files_info": all_file_info,  # Comprehensive file information
            "file_types": [info["file_type"] for info in all_file_info],
            "files_uploaded_to_storage": files_uploaded,
            "storage_upload_errors": upload_errors,
        }

        record = (
            await supabase_service.create_service_record(
                current_user.id, file_names, ai_response, metadata
            )
        ).get("record", None)

    return {
        "ai_response": ai_response,
        "share_id": record.get("share_id", None) if record else None,
    }


@visualization_router.get("/{share_id}")
async def get_visualization(share_id: str):
    """
    Get a visualization by its unique share_id

    Args:
        share_id: The unique share ID of the visualization

    Returns:
        Dict containing ai_response and share_id, or 404 if not found
    """
    try:
        # Get the service record by share_id
        record = await supabase_service.get_service_record_by_share_id(share_id)

        if not record:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=404,
                detail=f"Visualization with share_id '{share_id}' not found",
            )

        # Return the same format as /generate endpoint
        return {
            "ai_response": record.get("ai_response", {}),
            "share_id": record.get("share_id", share_id),
            "metadata": {
                "created_at": record.get("created_at"),
                "user_id": record.get("user_id"),
                "file_names": record.get("file_names", []),
                "processing_info": record.get("metadata", {}),
            },
        }

    except Exception as e:
        if "HTTPException" in str(type(e)):
            raise  # Re-raise HTTP exceptions

        from fastapi import HTTPException

        raise HTTPException(
            status_code=500, detail=f"Error retrieving visualization: {str(e)}"
        )


@system_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "document-processing-api",
    }

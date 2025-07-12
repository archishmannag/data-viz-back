from fastapi import APIRouter, UploadFile, Depends
from google import genai
from google.genai.types import Content, Part, GenerateContentConfig
from typing import Dict, Any, List
import fitz  # PyMuPDF
import re
import os
import json
import pandas as pd
import time
import math

import chunker
from supa import get_current_user_optional
from supabase_service import supabase_service

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Create router
processing_router = APIRouter(prefix="", tags=["document-processing"])
system_router = APIRouter(prefix="/system", tags=["system"])


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
    with open("response.json", "w") as f:
        f.write(str(response.text))
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
        text = page.get_text("text")
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
        print(f"Error processing Excel file: {str(e)}")
        # Return empty list on error instead of trying text processing
        return []


@processing_router.post("/generate")
async def generate_response(
    files: list[UploadFile], current_user=Depends(get_current_user_optional)
):
    """Generate a response based on the uploaded files.

    Args:
        files (list[UploadFile]): The files to process.
    """
    start_time = time.time()

    file_locations = []
    file_names = []
    file_sizes = []

    for i, file in enumerate(files):
        filename = file.filename or f"uploaded_file_{i}"
        file_location = f"/tmp/{filename}"
        file_locations.append(file_location)
        file_names.append(filename)

    # Save files and track sizes
    for i, file_location in enumerate(file_locations):
        content = await files[i].read()
        file_sizes.append(len(content))
        with open(file_location, "wb") as f:
            f.write(content)

    # Process files
    text_chunks: List[str] = []
    for file_location in file_locations:
        text_chunks.append(f"File: {file_location}\n")
        if file_location.endswith(".pdf"):
            text_chunks.extend(process_pdf(file_location))
        elif file_location.endswith(".xlsx") or file_location.endswith(".xls"):
            text_chunks.extend(process_excel(file_location))

    # Generate AI response
    var = generate_chart_response(text_chunks, len(files))

    # Calculate processing time
    processing_time = time.time() - start_time

    # Track service usage if user is authenticated
    if current_user:
        metadata = {
            "processing_time_seconds": processing_time,
            "file_sizes_bytes": file_sizes,
            "total_files": len(files),
            "endpoint": "/processing/generate",
            "file_types": [
                f.split(".")[-1] if "." in f else "unknown" for f in file_names
            ],
        }

        await supabase_service.create_service_record(
            current_user.id, file_names, var, metadata
        )

    return var


@system_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "document-processing-api",
    }

import asyncio
import io
import logging
import tempfile
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List

from .core import Screenshotter
from .exceptions import ScreenshotterError


logger = logging.getLogger(__name__)

# fastAPI app
app = FastAPI(
    title="Poetry Foundation Screenshotter API",
    description="Convert Poetry Foundation HTML files to PNG images by screenshotting poem content",
    version="0.1.0"
)

# add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# pydantic models
class ConversionRequest(BaseModel):
    """Request model for HTML to image conversion."""
    browser_type: str = Field(default="chromium", description="Browser type to use")
    headless: bool = Field(default=True, description="Run browser in headless mode")
    full_page: bool = Field(default=False, description="Take full page screenshot")


class ConversionResponse(BaseModel):
    """Response model for successful conversion."""
    message: str
    output_file: str
    input_file: str
    file_size_bytes: int


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None


class BatchConversionRequest(BaseModel):
    """Request model for batch conversion."""
    browser_type: str = Field(default="chromium", description="Browser type to use")
    headless: bool = Field(default=True, description="Run browser in headless mode")
    full_page: bool = Field(default=False, description="Take full page screenshot")
    parallel: int = Field(default=1, ge=1, le=5, description="Number of parallel conversions")


class BatchConversionResponse(BaseModel):
    """Response model for batch conversion."""
    message: str
    total_files: int
    successful_conversions: int
    failed_conversions: int
    results: List[dict]


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Screenshotter API",
        "version": "0.1.0",
        "description": "Convert HTML files to PNG images by screenshotting specific HTML content",
        "endpoints": {
            "convert": "/convert - Convert single HTML file to PNG",
            "batch_convert": "/batch-convert - Convert multiple HTML files to PNG",
            "health": "/health - Health check endpoint",
            "docs": "/docs - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "poetry-screenshotter"}


@app.post("/convert", response_model=ConversionResponse)
async def convert_html_to_png(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="HTML file to convert (.html or .html.gz)"),
    browser_type: str = Query("chromium", description="Browser type to use"),
    headless: bool = Query(True, description="Run browser in headless mode"),
    full_page: bool = Query(False, description="Take full page screenshot"),
):
    """Convert a single HTML file to PNG image.
    
    Upload an HTML file and get back a PNG image of the poem content.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    # validate file extension
    allowed_extensions = ['.html', '.gz']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail="File must be .html or .html.gz"
        )
    # create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='_' + file.filename) as temp_input:
        try:
            # write uploaded file to temp location
            content = await file.read()
            temp_input.write(content)
            temp_input.flush()
            input_path = Path(temp_input.name)
            # generate output filename
            if file.filename.lower().endswith('.html.gz'):
                output_filename = file.filename[:-8] + '.png'  # Remove .html.gz
            elif file.filename.lower().endswith('.html'):
                output_filename = file.filename[:-5] + '.png'  # Remove .html
            else:
                output_filename = file.filename + '.png'
            output_path = input_path.parent / output_filename
            # convert to PNG
            async with Screenshotter(headless=headless, browser_type=browser_type) as screenshotter:
                result_path = await screenshotter.convert_to_image(
                    input_path,
                    output_path,
                    full_page=full_page
                )
            # get file size
            file_size = result_path.stat().st_size
            # schedule cleanup
            def cleanup():
                try:
                    input_path.unlink(missing_ok=True)
                    result_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp files: {e}")
            background_tasks.add_task(cleanup)
            # return file response
            return FileResponse(
                path=str(result_path),
                filename=output_filename,
                media_type="image/png"
            )
            
        except ScreenshotterError as e:
            # cleanup on error
            try:
                input_path.unlink(missing_ok=True)
            except:
                pass
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # cleanup on error
            try:
                input_path.unlink(missing_ok=True)
            except:
                pass
            logger.error(f"Unexpected error in convert endpoint: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/batch-convert", response_model=BatchConversionResponse)
async def batch_convert_html_to_png(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="HTML files to convert"),
    browser_type: str = Query("chromium", description="Browser type to use"),
    headless: bool = Query(True, description="Run browser in headless mode"),
    full_page: bool = Query(False, description="Take full page screenshot"),
    parallel: int = Query(1, ge=1, le=5, description="Number of parallel conversions"),
):
    """Convert multiple HTML files to PNG images.
    
    Upload multiple HTML files and get back a zip file containing PNG images.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > 20: # you can adjust this limit as needed, we didn't think the use case would need more
        raise HTTPException(status_code=400, detail="Maximum 20 files allowed")
    temp_files = []
    results = []
    successful_conversions = 0
    try:
        # process each uploaded file
        for file in files:
            if not file.filename:
                results.append({"file": "unknown", "status": "error", "error": "No filename"})
                continue
            # validate file extension
            allowed_extensions = ['.html', '.gz']
            if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
                results.append({
                    "file": file.filename, 
                    "status": "error", 
                    "error": "Invalid file extension"
                })
                continue
            # save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='_' + file.filename) as temp_input:
                content = await file.read()
                temp_input.write(content)
                temp_input.flush()
                input_path = Path(temp_input.name)
                temp_files.append(input_path)
                # generate output path
                if file.filename.lower().endswith('.html.gz'):
                    output_filename = file.filename[:-8] + '.png'
                elif file.filename.lower().endswith('.html'):
                    output_filename = file.filename[:-5] + '.png'
                else:
                    output_filename = file.filename + '.png'
                output_path = input_path.parent / output_filename
                results.append({
                    "file": file.filename,
                    "input_path": input_path,
                    "output_path": output_path,
                    "output_filename": output_filename
                })
        
        # convert files
        async def convert_single(result_info):
            try:
                async with Screenshotter(headless=headless, browser_type=browser_type) as screenshotter:
                    await screenshotter.convert_to_image(
                        result_info["input_path"],
                        result_info["output_path"],
                        full_page=full_page
                    )
                result_info["status"] = "success"
                result_info["file_size"] = result_info["output_path"].stat().st_size
                return True
            except Exception as e:
                result_info["status"] = "error"
                result_info["error"] = str(e)
                return False
        
        # run conversions (with parallelism if requested)
        if parallel == 1:
            for result_info in results:
                if "input_path" in result_info:  # Skip already failed ones
                    if await convert_single(result_info):
                        successful_conversions += 1
        else:
            semaphore = asyncio.Semaphore(parallel)
            async def convert_with_semaphore(result_info):
                if "input_path" not in result_info:
                    return False
                async with semaphore:
                    return await convert_single(result_info)
            conversion_results = await asyncio.gather(
                *[convert_with_semaphore(r) for r in results],
                return_exceptions=True
            )
            successful_conversions = sum(1 for r in conversion_results if r is True)
        
        # create response
        response_results = []
        for result_info in results:
            response_result = {
                "file": result_info["file"],
                "status": result_info.get("status", "error")
            }
            if "error" in result_info:
                response_result["error"] = result_info["error"]
            if "file_size" in result_info:
                response_result["file_size_bytes"] = result_info["file_size"]
            response_results.append(response_result)
        
        # schedule cleanup
        def cleanup():
            for temp_file in temp_files:
                try:
                    temp_file.unlink(missing_ok=True)
                except:
                    pass
            for result_info in results:
                if "output_path" in result_info:
                    try:
                        result_info["output_path"].unlink(missing_ok=True)
                    except:
                        pass
        background_tasks.add_task(cleanup)
        
        return BatchConversionResponse(
            message=f"Batch conversion completed: {successful_conversions}/{len(files)} successful",
            total_files=len(files),
            successful_conversions=successful_conversions,
            failed_conversions=len(files) - successful_conversions,
            results=response_results
        )
        
    except Exception as e:
        # cleanup on error
        for temp_file in temp_files:
            try:
                temp_file.unlink(missing_ok=True)
            except:
                pass
        logger.error(f"Unexpected error in batch convert endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
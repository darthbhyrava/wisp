import asyncio
import click
import logging
import sys
import time

from pathlib import Path
from typing import Optional
from .core import Screenshotter
from .exceptions import ScreenshotterError

try:
    from tqdm.asyncio import tqdm as atqdm
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# configure logging
def setup_logging(verbose: bool = False, quiet_batch: bool = False):
    if quiet_batch:
        # for batch processing, only show warnings and errors
        level = logging.WARNING
        # suppress core module logging for cleaner output
        logging.getLogger('poetry_screenshotter.core').setLevel(logging.WARNING)
    else:
        level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(verbose)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option(
    '--output', '-o', 
    type=click.Path(path_type=Path),
    help='Output PNG file path (default: same as input with .png extension)'
)
@click.option(
    '--browser', '-b',
    type=click.Choice(['chromium', 'firefox', 'webkit']),
    default='chromium',
    help='Browser type to use for rendering'
)
@click.option(
    '--headless/--no-headless',
    default=True,
    help='Run browser in headless mode'
)
@click.option(
    '--full-page',
    is_flag=True,
    help='Take full page screenshot instead of just poem element'
)
@click.pass_context
def convert(ctx, input_file, output, browser, headless, full_page):
    """Convert a single HTML file to PNG image.
    
    INPUT_FILE: Path to the Poetry Foundation HTML file (.html or .html.gz)
    """
    verbose = ctx.obj.get('verbose', False)
    try:
        click.echo(f"Converting {input_file} to PNG...")
        screenshotter = Screenshotter(headless=headless, browser_type=browser)
        output_path = screenshotter.convert_to_image_sync(
            input_file, 
            output, 
            full_page=full_page
        )
        click.echo(f"Successfully saved screenshot to: {output_path}")
    except ScreenshotterError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        click.echo(f"Unexpected error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    help='Output directory for PNG files (default: same as input directory)'
)
@click.option(
    '--pattern', '-p',
    default='*.html.gz',
    help='File pattern to match (default: *.html.gz)'
)
@click.option(
    '--browser', '-b',
    type=click.Choice(['chromium', 'firefox', 'webkit']),
    default='chromium',
    help='Browser type to use for rendering'
)
@click.option(
    '--headless/--no-headless',
    default=True,
    help='Run browser in headless mode'
)
@click.option(
    '--full-page',
    is_flag=True,
    help='Take full page screenshot instead of just poem element'
)
@click.option(
    '--parallel', '-j',
    type=int,
    default=1,
    help='Number of parallel browser instances (default: 1)'
)
@click.pass_context
def batch(ctx, input_dir, output_dir, pattern, browser, headless, full_page, parallel):
    """Convert multiple HTML files to PNG images.
    
    INPUT_DIR: Directory containing Poetry Foundation HTML files
    """
    verbose = ctx.obj.get('verbose', False)
    # setup quieter logging for batch processing unless verbose is requested
    if not verbose:
        setup_logging(verbose=False, quiet_batch=True)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    # find matching files
    input_files = list(input_dir.glob(pattern))
    if not input_files:
        click.echo(f"No files found matching pattern '{pattern}' in {input_dir}")
        sys.exit(1)
    if not HAS_TQDM:
        click.echo("For better progress tracking, install tqdm: pip install tqdm")
        click.echo(f"Found {len(input_files)} files to convert...")
    
    async def convert_file(file_path: Path, screenshotter: Screenshotter, pbar=None):
        """convert a single file."""
        try:
            # generate output path
            if file_path.suffix == '.gz':
                output_name = file_path.with_suffix('').with_suffix('.png').name
            else:
                output_name = file_path.with_suffix('.png').name
            output_path = output_dir / output_name
            result = await screenshotter.convert_to_image(
                file_path,
                output_path,
                full_page=full_page
            )
            if pbar:
                pbar.set_description(f"✅ {file_path.name}")
                pbar.update(1)
            elif verbose:
                click.echo(f"{file_path.name} -> {result.name}")
            return True
        except Exception as e:
            error_msg = f"{file_path.name}: {str(e)}"
            if pbar:
                pbar.set_description(f"❌ {file_path.name}")
                pbar.update(1)
                # Write error to stderr so it doesn't interfere with progress bar
                click.echo(error_msg, err=True)
            else:
                click.echo(error_msg)
            return False
    
    async def run_batch():
        """Run batch conversion."""
        success_count = 0
        start_time = time.time()
        if HAS_TQDM:
            pbar = tqdm(
                total=len(input_files),
                desc="Converting",
                unit="file",
                ncols=80,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
        else:
            pbar = None
        try:
            if parallel == 1:
                # Sequential processing
                async with Screenshotter(headless=headless, browser_type=browser) as screenshotter:
                    for file_path in input_files:
                        if await convert_file(file_path, screenshotter, pbar):
                            success_count += 1
            else:
                # parallel processing with semaphore
                semaphore = asyncio.Semaphore(parallel)
                async def convert_with_semaphore(file_path: Path):
                    async with semaphore:
                        async with Screenshotter(headless=headless, browser_type=browser) as screenshotter:
                            return await convert_file(file_path, screenshotter, pbar)
                results = await asyncio.gather(
                    *[convert_with_semaphore(f) for f in input_files],
                    return_exceptions=True
                )
                success_count = sum(1 for r in results if r is True)
            if pbar:
                pbar.close()
            # final summary
            elapsed = time.time() - start_time
            if success_count == len(input_files):
                click.echo(f"✅ All {success_count} files converted successfully in {elapsed:.1f}s")
            else:
                failed_count = len(input_files) - success_count
                click.echo(f"Batch conversion complete: {success_count}/{len(input_files)} successful, {failed_count} failed in {elapsed:.1f}s")
        except Exception as e:
            if pbar:
                pbar.close()
            raise e
        if success_count < len(input_files):
            sys.exit(1)
    try:
        asyncio.run(run_batch())
    except KeyboardInterrupt:
        click.echo("\nConversion cancelled by user")
        sys.exit(1)
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
        click.echo(f"Batch conversion failed: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option(
    '--port', '-p',
    type=int,
    default=8000,
    help='Port to run the API server on (default: 8000)'
)
@click.option(
    '--host',
    default='127.0.0.1',
    help='Host to bind the server to (default: 127.0.0.1)'
)
@click.option(
    '--reload',
    is_flag=True,
    help='Enable auto-reload for development'
)
def serve(port, host, reload):
    """Start the FastAPI server for API access."""
    try:
        import uvicorn
        from .api import app
        click.echo(f"Starting poetry-screenshotter API server on {host}:{port}")
        uvicorn.run(
            "poetry_screenshotter.api:app",
            host=host,
            port=port,
            reload=reload
        )
    except ImportError:
        click.echo("FastAPI/uvicorn not installed. Install with: pip install 'poetry-screenshotter[api]'")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Failed to start server: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
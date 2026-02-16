"""
Command-line interface for VidSubX subtitle extractor.

Usage:
    python cmd.py <video_path> [options]

Arguments:
    video_path               Path to the video file for subtitle extraction.

Options:
    --output OUTPUT          Output SRT file path (default: same as video with .srt extension).
    --subtitle_area X1 Y1 X2 Y2
                            Subtitle area coordinates in pixels (x1, y1, x2, y2).
    --start_frame FRAME     Start extraction from this frame number.
    --stop_frame FRAME      Stop extraction at this frame number.
    --language LANG         OCR recognition language code (default: ch).
    --frame_freq FREQ       Frame extraction frequency (default: 2).
    --text_batch_size SIZE  Text extraction batch size (default: 100).
    --use_mobile_model      Use mobile OCR model (faster, less accurate).
    --use_gpu               Enable GPU acceleration if available.
    --gpu_provider PROVIDER GPU provider: cuda, directml, openvino (default: cuda).
    --cpu_processes NUM     Number of CPU processes for OCR (default: half of physical cores).
    --gpu_processes NUM     Number of GPU processes for OCR (default: one third of physical cores).
    --log_file LOGFILE      Log file path (default: temp log file in system temp directory).
    --quiet                 Suppress console output.
    --version               Show version and exit.
    --help                  Show this help message and exit.

Examples:
    python cmd.py "video.mp4" --language en --output "output.srt"
    python cmd.py "video.mp4" --subtitle_area 0 900 1920 1080 --start_frame 1000 --stop_frame 5000
    python cmd.py "video.mp4" --use_gpu --gpu_provider cuda --cpu_processes 4
"""

import argparse
import logging
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from main import SubtitleExtractor, setup_ocr
from utilities.logger_setup import setup_logging
from utilities.utils import CONFIG


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract hardcoded subtitles from video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1].strip()
    )

    # Required arguments
    parser.add_argument(
        "video_path",
        help="Path to the video file for subtitle extraction."
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output SRT file path (default: same as video with .srt extension)."
    )

    # Subtitle area options
    parser.add_argument(
        "--subtitle_area", "-s",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Subtitle area coordinates in pixels (x1, y1, x2, y2)."
    )

    # Frame range options
    parser.add_argument(
        "--start_frame",
        type=int,
        help="Start extraction from this frame number."
    )
    parser.add_argument(
        "--stop_frame",
        type=int,
        help="Stop extraction at this frame number."
    )

    # OCR options
    parser.add_argument(
        "--language", "-l",
        default="ch",
        help="OCR recognition language code (default: ch)."
    )
    parser.add_argument(
        "--frame_freq",
        type=int,
        default=2,
        help="Frame extraction frequency (default: 2)."
    )
    parser.add_argument(
        "--text_batch_size",
        type=int,
        default=100,
        help="Text extraction batch size (default: 100)."
    )
    parser.add_argument(
        "--use_mobile_model",
        action="store_true",
        help="Use mobile OCR model (faster, less accurate)."
    )
    parser.add_argument(
        "--text_drop_score",
        type=float,
        default=0.6,
        help="Text drop confidence threshold (default: 0.6)."
    )
    parser.add_argument(
        "--no_line_break",
        action="store_true",
        help="Don't use line breaks in extracted text."
    )

    # Performance options
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Enable GPU acceleration if available."
    )
    parser.add_argument(
        "--gpu_provider",
        choices=["cuda", "directml", "openvino"],
        default="cuda",
        help="GPU provider: cuda, directml, openvino (default: cuda)."
    )
    parser.add_argument(
        "--cpu_processes",
        type=int,
        help="Number of CPU processes for OCR (default: half of physical cores)."
    )
    parser.add_argument(
        "--gpu_processes",
        type=int,
        help="Number of GPU processes for OCR (default: one third of physical cores)."
    )
    parser.add_argument(
        "--cpu_onnx_threads",
        type=int,
        help="Number of ONNX intra-op threads for CPU (default: based on core count)."
    )

    # Subtitle processing options
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.85,
        help="Text similarity threshold for merging (default: 0.85)."
    )
    parser.add_argument(
        "--min_sub_duration",
        type=float,
        default=120.0,
        help="Minimum subtitle duration in milliseconds (default: 120.0)."
    )
    parser.add_argument(
        "--min_consecutive_duration",
        type=float,
        default=500.0,
        help="Minimum consecutive subtitle duration in ms (default: 500.0)."
    )
    parser.add_argument(
        "--max_consecutive_short",
        type=int,
        default=4,
        help="Max consecutive short durations to keep (default: 4)."
    )

    # Logging and output options
    parser.add_argument(
        "--log_file",
        help="Log file path (default: temp log file in system temp directory)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output."
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit."
    )

    return parser.parse_args()


def setup_logging_cli(log_file=None, quiet=False):
    """Setup logging for command line interface."""
    if quiet:
        # Disable all logging except errors
        logging.getLogger().setLevel(logging.ERROR)
    else:
        # Setup normal logging
        setup_logging()
    
    # If log file specified, add file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def update_config_from_args(args):
    """Update CONFIG from command line arguments."""
    # OCR Engine settings
    CONFIG.ocr_rec_language = args.language
    CONFIG.use_mobile_model = args.use_mobile_model
    CONFIG.gpu_provider = args.gpu_provider
    
    # Performance settings
    CONFIG.use_gpu = args.use_gpu
    if args.cpu_processes:
        CONFIG.cpu_ocr_processes = args.cpu_processes
    if args.gpu_processes:
        CONFIG.gpu_ocr_processes = args.gpu_processes
    if args.cpu_onnx_threads:
        CONFIG.cpu_onnx_intra_threads = args.cpu_onnx_threads
    
    # Frame extraction settings
    CONFIG.frame_extraction_frequency = args.frame_freq
    CONFIG.text_extraction_batch_size = args.text_batch_size
    CONFIG.text_drop_score = args.text_drop_score
    CONFIG.line_break = not args.no_line_break
    
    # Subtitle processing settings
    CONFIG.text_similarity_threshold = args.similarity_threshold
    CONFIG.min_sub_duration_ms = args.min_sub_duration
    CONFIG.min_consecutive_sub_dur_ms = args.min_consecutive_duration
    CONFIG.max_consecutive_short_durs = args.max_consecutive_short
    
    # Update OCR options
    CONFIG.ocr_opts["lang"] = CONFIG.ocr_rec_language
    CONFIG.ocr_opts["use_mobile_model"] = CONFIG.use_mobile_model
    CONFIG.ocr_opts["use_textline_orientation"] = CONFIG.use_text_ori


def print_version():
    """Print version information."""
    print("VidSubX - Video Subtitle Extractor")
    print("Version: 1.0.0")
    print("Author: VidSubX Team")
    print("License: MIT")
    print("\nFeatures:")
    print("  - Hardcoded subtitle extraction")
    print("  - Multiple OCR language support")
    print("  - GPU acceleration (CUDA, DirectML, OpenVINO)")
    print("  - Batch processing")
    print("  - Automatic subtitle area detection")


def main():
    """Main command line interface function."""
    args = parse_arguments()
    
    # Handle version flag
    if args.version:
        print_version()
        return 0
    
    # Check if video file exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        return 1
    
    # Setup logging
    if not args.log_file:
        # Create temporary log file
        temp_dir = Path(tempfile.gettempdir())
        log_file = temp_dir / f"{video_path.stem}_subtitle_extraction.log"
    else:
        log_file = args.log_file
    
    setup_logging_cli(log_file, args.quiet)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting subtitle extraction for: {video_path.name}")
    logger.info(f"Log file: {log_file}")
    
    # Update configuration from command line arguments
    update_config_from_args(args)
    
    # Setup OCR
    try:
        setup_ocr()
    except Exception as e:
        logger.error(f"Failed to setup OCR: {e}", exc_info=True)
        print(f"Error: Failed to setup OCR: {e}", file=sys.stderr)
        return 1
    
    # Prepare subtitle area if provided
    subtitle_area = None
    if args.subtitle_area:
        subtitle_area = tuple(args.subtitle_area)
        logger.info(f"Using custom subtitle area: {subtitle_area}")
    
    # Create subtitle extractor instance
    extractor = SubtitleExtractor()
    
    # Run extraction
    try:
        print(f"Extracting subtitles from {video_path.name}...")
        if not args.quiet:
            print(f"Language: {args.language}")
            print(f"GPU Acceleration: {'Yes' if args.use_gpu else 'No'} ({args.gpu_provider.upper()})")
            print(f"Log file: {log_file}")
        
        output_path = extractor.run_extraction(
            video_path=str(video_path),
            sub_area=subtitle_area,
            start_frame=args.start_frame,
            stop_frame=args.stop_frame
        )
        
        if output_path:
            print(f"\nSubtitle extraction completed successfully!")
            print(f"Output file: {output_path}")
            
            # If output path was specified, rename/move the file
            if args.output:
                target_path = Path(args.output)
                output_path.rename(target_path)
                print(f"Moved to: {target_path}")
            
            logger.info(f"Subtitle saved to: {output_path}")
            return 0
        else:
            print("\nSubtitle extraction failed to produce output.", file=sys.stderr)
            logger.error("Subtitle extraction failed to produce output.")
            return 1
            
    except KeyboardInterrupt:
        print("\nSubtitle extraction cancelled by user.")
        logger.warning("Subtitle extraction cancelled by user.")
        return 130
    except Exception as e:
        print(f"\nError during subtitle extraction: {e}", file=sys.stderr)
        logger.error(f"Error during subtitle extraction: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""
Progress tracking module for the statapp application.

This module provides a class for tracking progress of file operations
with dual progress bars (overall and per-file).
"""

import threading
import time
from threading import Lock
from typing import List, Callable, Any

import humanize
from rich.live import Live

from statapp.utils.utils import console, create_dual_progress


class ProgressTracker:
    """
    Class for tracking progress of file operations with dual progress bars.

    This class provides a way to track progress of file operations with two progress bars:
    - An overall progress bar showing total progress across all files
    - A per-file progress bar showing progress for the current file

    It handles thread-safe updates to the progress bars and provides a clean interface
    for updating progress during file operations.
    """

    def __init__(self, files: List[Any], get_file_size: Callable[[Any], int], total_files: int = None):
        """
        Initialize the progress tracker.

        Args:
            files: List of files to process
            get_file_size: Function to get the size of a file
            total_files: Total number of files to process (if different from len(files))
        """
        self.files = files
        self.get_file_size = get_file_size
        self.progress = create_dual_progress()
        self.total_files = total_files if total_files is not None else len(files)

        # Calculate total size
        self.total_size = sum(get_file_size(file) for file in files)
        self.total_size_human = humanize.naturalsize(self.total_size, binary=True)

        # Initialize progress tracking variables
        self.uploaded_size = 0
        self.current_file_progress = 0
        self.current_file_size = 0
        self.current_file_uploaded = 0  # Track accumulated bytes for current file
        self.total_start_time = 0

        # Thread-safe counter for completed files
        self.files_completed_count = 0
        self.files_completed_lock = Lock()

        # Flag to control the background thread
        self.stop_thread = threading.Event()
        self.progress_thread = None

        # Progress bar task IDs
        self.overall_task = None
        self.file_task = None

        # Live context manager for displaying progress
        self.live = None

    def start(self) -> None:
        """
        Start tracking progress.

        This method initializes the progress bars and starts the background thread
        for updating the overall progress bar.
        """
        # Add task for overall progress
        self.overall_task = self.progress.add_task(
            "[bold blue]Overall Progress", 
            total=100,  # Use 100 for percentage-based progress
            size=f"0/{self.total_size_human}",
            elapsed="0s"
        )

        # Add task for file progress
        self.file_task = self.progress.add_task(
            "[bold yellow]Current File",
            total=100,  # Percentage
            size="0 B",
            elapsed="0s",
            visible=False  # Hide initially
        )

        # Record start time
        self.total_start_time = time.time()

        # Create and start the Live context manager
        self.live = Live(self.progress, console=console, refresh_per_second=2, transient=True)
        self.live.start()

        # Start background thread for updating overall progress
        self.progress_thread = threading.Thread(target=self._update_overall_progress)
        self.progress_thread.daemon = True  # Thread will exit when main thread exits
        self.progress_thread.start()

    def stop(self) -> None:
        """
        Stop tracking progress.

        This method stops the background thread for updating the overall progress bar
        and stops the Live context manager.
        """
        # Stop the background thread
        self.stop_thread.set()
        if self.progress_thread and self.progress_thread.is_alive():
            self.progress_thread.join(timeout=2)  # Wait for the thread to finish, but not indefinitely

        # Stop the Live context manager
        if self.live:
            self.live.stop()

    def start_file(self, file: Any, display_path: str, file_size: int) -> None:
        """
        Start tracking progress for a file.

        Args:
            file: The file being processed
            display_path: The path to display in the progress bar
            file_size: The size of the file in bytes
        """
        # Calculate elapsed time for overall progress
        current_total_elapsed = time.time() - self.total_start_time
        total_elapsed_str = f"{int(current_total_elapsed)}s"

        # Update overall progress display with thread-safe counter
        with self.files_completed_lock:
            current_files_completed = self.files_completed_count

        self.progress.update(
            self.overall_task, 
            description=f"[bold blue]Overall Progress ({current_files_completed}/{self.total_files} files)",
            size=f"{humanize.naturalsize(self.uploaded_size, binary=True)}/{self.total_size_human}",
            elapsed=total_elapsed_str
        )

        # Update and show file progress
        file_size_human = humanize.naturalsize(file_size, binary=True)
        self.progress.update(
            self.file_task,
            completed=0,
            description=f"[bold yellow]Downloading: {display_path}",
            size=file_size_human,
            elapsed="0s",
            visible=True
        )

        # Update current file size for overall progress calculation
        self.current_file_size = file_size
        self.current_file_progress = 0
        self.current_file_uploaded = 0  # Reset accumulated bytes for new file

    def update_file_progress(self, bytes_transferred: int, file_size: int, display_path: str, start_time: float) -> None:
        """
        Update progress for the current file.

        Args:
            bytes_transferred: Number of bytes in this chunk
            file_size: Total size of the file in bytes
            display_path: The path to display in the progress bar
            start_time: Time when the file transfer started
        """
        if file_size > 0:
            # Accumulate bytes transferred
            self.current_file_uploaded += bytes_transferred

            # Calculate percentage based on accumulated bytes
            percent_complete = min(100, int(self.current_file_uploaded / file_size * 100))

            # Update current file progress for overall progress calculation
            self.current_file_progress = percent_complete

            # Calculate elapsed time
            current_time = time.time()
            elapsed = current_time - start_time
            elapsed_str = f"{int(elapsed)}s"

            # Update progress bar
            self.progress.update(
                self.file_task,
                completed=percent_complete,
                description=f"[bold yellow]Downloading: {display_path} ({percent_complete}%)",
                elapsed=elapsed_str
            )

    def complete_file(self, display_path: str, file_size: int, start_time: float, success: bool = True) -> None:
        """
        Mark a file as completed.

        Args:
            display_path: The path to display in the progress bar
            file_size: The size of the file in bytes
            start_time: Time when the file transfer started
            success: Whether the file was processed successfully
        """
        end_time = time.time()

        if success:
            # Update progress after upload completes
            self.uploaded_size += file_size
            elapsed = end_time - start_time
            total_elapsed = end_time - self.total_start_time

            # Reset current file progress, size, and uploaded bytes
            self.current_file_progress = 0
            self.current_file_size = 0
            self.current_file_uploaded = 0

            # Calculate elapsed times
            file_elapsed_str = f"{int(elapsed)}s"
            total_elapsed_str = f"{int(total_elapsed)}s"

            # Update file progress to show completion (ensure it's at 100%)
            self.progress.update(
                self.file_task,
                completed=100,
                description=f"[bold green]Downloaded: {display_path}",
                elapsed=file_elapsed_str
            )
        else:
            # Calculate elapsed time for overall progress
            current_total_elapsed = time.time() - self.total_start_time
            total_elapsed_str = f"{int(current_total_elapsed)}s"

            # Reset current file progress, size, and uploaded bytes
            self.current_file_progress = 0
            self.current_file_size = 0
            self.current_file_uploaded = 0

            # Update file progress to show error
            self.progress.update(
                self.file_task,
                completed=0,
                description=f"[bold red]Error: {display_path}",
                size="Error",
                elapsed="N/A"
            )

        # Update the thread-safe counter
        with self.files_completed_lock:
            self.files_completed_count += 1
            current_files_completed = self.files_completed_count

        # Update overall progress display
        self.progress.update(
            self.overall_task, 
            description=f"[bold blue]Overall Progress ({current_files_completed}/{self.total_files} files)",
            size=f"{humanize.naturalsize(self.uploaded_size, binary=True)}/{self.total_size_human}",
            elapsed=total_elapsed_str
        )

    def get_progress_callback(self, display_path: str, file_size: int, start_time: float) -> Callable[[int], None]:
        """
        Get a callback function for updating progress during file transfer.

        Args:
            display_path: The path to display in the progress bar
            file_size: The size of the file in bytes
            start_time: Time when the file transfer started

        Returns:
            A callback function that can be passed to file transfer functions
        """
        def callback(bytes_transferred: int) -> None:
            self.update_file_progress(bytes_transferred, file_size, display_path, start_time)

        return callback

    def _update_overall_progress(self) -> None:
        """
        Update the overall progress bar every second.

        This method runs in a background thread and updates the overall progress bar
        with the current progress, including the progress of the current file.
        """
        while not self.stop_thread.is_set():
            # Calculate elapsed time for overall progress
            current_total_elapsed = time.time() - self.total_start_time
            total_elapsed_str = f"{int(current_total_elapsed)}s"

            # Calculate overall progress including current file's progress
            current_progress_bytes = 0
            if self.current_file_size > 0:
                current_progress_bytes = int(self.current_file_size * self.current_file_progress / 100)

            total_progress_bytes = self.uploaded_size + current_progress_bytes
            total_progress_percent = min(100, int(total_progress_bytes / self.total_size * 100)) if self.total_size > 0 else 0

            # Get the current count of completed files (thread-safe)
            with self.files_completed_lock:
                current_files_completed = self.files_completed_count

            # Update overall progress display
            self.progress.update(
                self.overall_task, 
                completed=total_progress_percent,
                description=f"[bold blue]Overall Progress ({current_files_completed}/{self.total_files} files)",
                size=f"{humanize.naturalsize(total_progress_bytes, binary=True)}/{self.total_size_human}",
                elapsed=total_elapsed_str
            )

            # Sleep for 0.5 seconds to match refresh rate of 2 updates per second
            time.sleep(0.5)

def track_progress(files: List[Any], get_file_size: Callable[[Any], int], process_file: Callable[[Any, ProgressTracker], None], total_files: int = None) -> None:
    """
    Track progress of processing a list of files.

    Args:
        files: List of files to process
        get_file_size: Function to get the size of a file
        process_file: Function to process a file, takes a file and a ProgressTracker
        total_files: Total number of files to process (if different from len(files))
    """
    # Create progress tracker
    tracker = ProgressTracker(files, get_file_size, total_files)

    # Start tracking progress (this now creates its own Live context manager)
    tracker.start()

    try:
        # Process each file
        for file in files:
            process_file(file, tracker)
    finally:
        # Stop tracking progress (this now stops the Live context manager)
        tracker.stop()

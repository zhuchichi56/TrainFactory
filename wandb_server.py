#!/usr/bin/env python3
"""
WandB offline data scheduled synchronization script
Supports scheduled synchronization of offline cached runs to online service
"""

import os
import time
import subprocess
import schedule
import logging
from datetime import datetime
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wandb_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WandbSyncer:
    def __init__(self, wandb_dir=None):
        """
        Initialize WandB syncer

        Args:
            wandb_dir: wandb cache directory path, defaults to ./wandb
        """
        self.wandb_dir = Path(wandb_dir) if wandb_dir else Path("./wandb")
        self.offline_runs_dir = self.wandb_dir / "offline-run-*"

    def find_offline_runs(self):
        """Find all offline run directories"""
        offline_runs = list(self.wandb_dir.glob("offline-run-*"))
        logger.info(f"Found {len(offline_runs)} offline runs")
        return offline_runs

    def sync_single_run(self, run_dir):
        """Sync a single run to online"""
        try:
            logger.info(f"Starting sync for run: {run_dir.name}")

            # Use wandb sync command to synchronize
            result = subprocess.run(
                ["wandb", "sync", str(run_dir)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"Successfully synced run: {run_dir.name}")
                return True
            else:
                logger.error(f"Sync failed for {run_dir.name}: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Sync timeout for: {run_dir.name}")
            return False
        except Exception as e:
            logger.error(f"Sync error for {run_dir.name}: {str(e)}")
            return False

    def sync_all_offline_runs(self, delete_after_sync=False):
        """Sync all offline runs"""
        offline_runs = self.find_offline_runs()

        if not offline_runs:
            logger.info("No offline runs found, skipping sync")
            return

        success_count = 0
        failed_count = 0

        for run_dir in offline_runs:
            if self.sync_single_run(run_dir):
                success_count += 1

                # Optional: delete local cache after successful sync
                if delete_after_sync:
                    try:
                        import shutil
                        shutil.rmtree(run_dir)
                        logger.info(f"Deleted local cache: {run_dir.name}")
                    except Exception as e:
                        logger.error(f"Failed to delete cache {run_dir.name}: {str(e)}")
            else:
                failed_count += 1

        logger.info(f"Sync completed - Success: {success_count}, Failed: {failed_count}")

    def check_wandb_login(self):
        """Check if wandb is logged in"""
        try:
            # Method 1: Check for WANDB_API_KEY environment variable
            api_key = os.environ.get('WANDB_API_KEY')
            if api_key:
                logger.info("Found WANDB_API_KEY in environment variables")
                # Try to use the API key to verify authentication
                try:
                    import wandb
                    # For older versions of wandb, try setting the API key directly
                    if hasattr(wandb, 'login'):
                        wandb.login(key=api_key, relogin=True)
                    else:
                        # For older versions, set environment and create API instance
                        os.environ['WANDB_API_KEY'] = api_key

                    api = wandb.Api()
                    user = api.viewer
                    if user:
                        username = getattr(user, 'username', getattr(user, 'name', 'unknown'))
                        logger.info(f"WandB is authenticated as: {username}")
                        return True
                except Exception as e:
                    logger.info(f"API authentication attempt failed: {str(e)}")
                    # If API method fails, the environment variable is still set for subprocess calls
                    logger.info("WANDB_API_KEY is set, proceeding with sync (auth will be handled by wandb sync command)")
                    return True

            # Method 2: Try to import wandb and check if logged in programmatically
            try:
                import wandb
                api = wandb.Api()
                user = api.viewer
                if user:
                    username = getattr(user, 'username', getattr(user, 'name', 'unknown'))
                    logger.info(f"WandB is logged in as: {username}")
                    return True
            except Exception as e:
                logger.info(f"Programmatic check failed: {str(e)}")

            # Method 3: Check for wandb settings file
            wandb_dir = os.path.expanduser("~/.config/wandb")
            settings_file = os.path.join(wandb_dir, "settings")
            netrc_file = os.path.expanduser("~/.netrc")

            if os.path.exists(settings_file) or os.path.exists(netrc_file):
                logger.info("WandB credentials found in config files")
                return True

            logger.error("WandB not authenticated. Please either:")
            logger.error("1. Set WANDB_API_KEY environment variable: export WANDB_API_KEY=your_api_key")
            logger.error("2. Run 'wandb login' command")
            return False

        except Exception as e:
            logger.error(f"Failed to check login status: {str(e)}")
            return False

def sync_job(syncer, delete_after_sync=False):
    """Scheduled sync job execution"""
    logger.info("=" * 50)
    logger.info("Starting scheduled sync task")

    if not syncer.check_wandb_login():
        logger.error("WandB not logged in, skipping sync")
        return

    syncer.sync_all_offline_runs(delete_after_sync=delete_after_sync)
    logger.info("Scheduled sync task completed")
    logger.info("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='WandB offline data scheduled sync tool')
    parser.add_argument('--wandb-dir', default="./wandb",
                       help='WandB cache directory path (default: ./wandb)')
    parser.add_argument('--interval', type=int, default=1,
                       help='Sync interval in minutes (default: 5)')
    parser.add_argument('--delete-after-sync', action='store_true',
                       help='Delete local cache after successful sync')
    parser.add_argument('--sync-once', action='store_true',
                       help='Execute sync only once, do not start scheduled task')
    parser.add_argument('--schedule-time', default=None,
                       help='Daily scheduled sync time, format like "14:30" (mutually exclusive with --interval)')

    args = parser.parse_args()

    # Create syncer
    syncer = WandbSyncer(wandb_dir=args.wandb_dir)

    if args.sync_once:
        # Execute sync only once
        sync_job(syncer, delete_after_sync=args.delete_after_sync)
        return

    # Set up scheduled tasks
    if args.schedule_time:
        # Daily sync at specified time
        schedule.every().day.at(args.schedule_time).do(
            sync_job, syncer, args.delete_after_sync
        )
        logger.info(f"Set daily sync at {args.schedule_time}")
    else:
        # Sync at interval
        schedule.every(args.interval).minutes.do(
            sync_job, syncer, args.delete_after_sync
        )
        logger.info(f"Set sync every {args.interval} minutes")

    logger.info("Scheduled sync service started, press Ctrl+C to stop")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received stop signal, exiting...")

if __name__ == "__main__":
    main()

"""
Usage:

1. Install dependencies:
   pip install wandb schedule

2. Login to wandb:
   wandb login

3. Basic usage:
   export WANDB_API_KEY=xxx
   python wandb_server.py                          # Sync every 5 minutes
   python wandb_server.py --interval 30            # Sync every 30 minutes
   python wandb_server.py --schedule-time "14:30"  # Sync daily at 2:30 PM
   python wandb_server.py --delete-after-sync      # Delete local cache after successful sync
   python wandb_server.py --sync-once              # Execute sync only once
   python wandb_server.py --wandb-dir /custom/path # Specify wandb directory

4. Combined usage:
   python wandb_server.py --interval 30 --delete-after-sync
   python scripts/train/wandb.py --schedule-time "02:00" --delete-after-sync
"""
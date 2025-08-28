#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Background Training Monitoring Script
Monitor training process status, resource usage, and progress
"""

import os
import time
import subprocess
import sys
from datetime import datetime
import re

def get_training_process():
    """Check if training process is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'train_gender_bias_model.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pid = result.stdout.strip()
            return pid
        return None
    except:
        return None

def get_cpu_usage(pid):
    """Get process CPU usage"""
    try:
        result = subprocess.run(['ps', '-p', pid, '-o', '%cpu'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return float(lines[1])
        return 0
    except:
        return 0

def get_memory_usage(pid):
    """Get process memory usage"""
    try:
        result = subprocess.run(['ps', '-p', pid, '-o', 'rss'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return int(lines[1]) / 1024  # Convert to MB
        return 0
    except:
        return 0

def parse_log_progress():
    """Parse training log to get progress"""
    log_files = ['training_output.log', 'gender_bias_training.log', 'fast_training.log']
    
    current_epoch = 0
    total_epochs = 30
    batch_progress = ""
    loss_info = ""
    
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Search from bottom up for latest information
                for line in reversed(lines[-50:]):  # Only check last 50 lines
                    # Find epoch information
                    if 'Epoch ' in line and '/' in line:
                        match = re.search(r'Epoch (\d+)/(\d+)', line)
                        if match:
                            current_epoch = int(match.group(1))
                            total_epochs = int(match.group(2))
                    
                    # Find training progress
                    if 'Training:' in line and '%' in line:
                        batch_progress = line.strip().split('Training:')[1].strip()
                    elif 'Training' in line and '%' in line:
                        batch_progress = line.strip()
                    
                    # Find loss information
                    if 'loss=' in line:
                        match = re.search(r'loss=([\d.]+)', line)
                        if match:
                            loss_info = f"Loss: {match.group(1)}"
                    elif 'Loss:' in line:
                        match = re.search(r'Loss: ([\d.]+)', line)
                        if match:
                            loss_info = f"Loss: {match.group(1)}"
                
                break
            except:
                continue
    
    return current_epoch, total_epochs, batch_progress, loss_info

def show_status():
    """Display training status"""
    print("🔍 Instagram Gender Bias Model Training Status")
    print("=" * 50)
    
    # Check process
    pid = get_training_process()
    if pid:
        print(f"✅ Training process running (PID: {pid})")
        
        # Get resource usage
        cpu_usage = get_cpu_usage(pid)
        memory_usage = get_memory_usage(pid)
        print(f"📊 CPU Usage: {cpu_usage:.1f}%")
        print(f"💾 Memory Usage: {memory_usage:.1f} MB")
        
        # Get training progress
        current_epoch, total_epochs, batch_progress, loss_info = parse_log_progress()
        
        if current_epoch > 0:
            epoch_progress = (current_epoch / total_epochs) * 100
            print(f"📈 Training Progress: Epoch {current_epoch}/{total_epochs} ({epoch_progress:.1f}%)")
            
            if batch_progress:
                print(f"📋 Current Batch: {batch_progress}")
            
            if loss_info:
                print(f"📉 {loss_info}")
        else:
            print("⏳ Training is initializing...")
        
    else:
        print("❌ Training process not running")
        
        # Check for log files
        log_files = ['training_output.log', 'fast_training.log', 'gender_bias_training.log']
        for log_file in log_files:
            if os.path.exists(log_file):
                print(f"📄 Checking log file: {log_file}...")
                with open(log_file, 'r') as f:
                    last_lines = f.readlines()[-10:]
                
                for line in last_lines:
                    if 'ERROR' in line or 'Traceback' in line:
                        print(f"❌ Error found, please check {log_file}")
                        break
                break
    
    print("=" * 50)
    print(f"⏰ Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show quick commands
    print("\n💡 Quick Commands:")
    print("  View real-time log: tail -f fast_training.log")
    print("  View detailed log: tail -f training_output.log") 
    print("  Stop training: pkill -f train_fast_local")
    print("  Restart training: nohup python train_fast_local_english.py > training_output.log 2>&1 &")

def monitor_continuously():
    """Continuous monitoring mode"""
    try:
        while True:
            os.system('clear')  # Clear screen
            show_status()
            
            pid = get_training_process()
            if not pid:
                print("\n❌ Training process has stopped, exiting monitor")
                break
                
            print("\nPress Ctrl+C to exit monitoring")
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Exiting monitor")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == 'watch':
        monitor_continuously()
    else:
        show_status()

if __name__ == "__main__":
    main()

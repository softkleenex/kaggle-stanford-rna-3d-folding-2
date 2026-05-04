#!/bin/bash
# The Ultimate AI Agentic Loop
# This script spawns 'gemini' CLI in headless (-y) mode to autonomously write code, fix errors, and do research!

COMPETITION="stanford-rna-3d-folding-2"
VERSION=27
CURRENT_KERNEL="softkleenex/rna-folding-v27-top1-fixed"
LOG_FILE="ai_loop.log"

echo "=== STARTING GEMINI AGENTIC LOOP ===" > $LOG_FILE

while true; do
    echo "$(date) | Monitoring $CURRENT_KERNEL..." | tee -a $LOG_FILE
    
    while true; do
        STATUS=$(kaggle kernels status $CURRENT_KERNEL 2>/dev/null)
        echo "$(date) | Status: $STATUS" >> $LOG_FILE
        
        if [[ $STATUS == *"COMPLETE"* ]]; then
            echo "$(date) | ✅ COMPLETE! Submitting to leaderboard..." | tee -a $LOG_FILE
            mkdir -p submissions/v$VERSION
            kaggle kernels output $CURRENT_KERNEL -p submissions/v$VERSION --force
            kaggle competitions submit -c $COMPETITION -k $CURRENT_KERNEL -f submission.csv -m "Auto Submit v$VERSION" -v 1
            
            # Wait for score
            while true; do
                SUB_STATUS=$(kaggle competitions submissions -c $COMPETITION --csv | head -n 2 | tail -n 1)
                if [[ $SUB_STATUS == *"pending"* ]] || [[ $SUB_STATUS == *"scoring"* ]]; then
                    sleep 60
                else
                    SCORE=$(echo "$SUB_STATUS" | awk -F',' '{print $5}')
                    echo "$(date) | 🎯 SCORE: $SCORE" | tee -a $LOG_FILE
                    
                    NEXT_VER=$((VERSION + 1))
                    echo "🧠 Invoking Gemini AI for Architectural Upgrade to v$NEXT_VER..." | tee -a $LOG_FILE
                    
                    # Call Gemini in headless mode to do research and write the next script
                    gemini -y -p "The Kaggle kernel $CURRENT_KERNEL scored $SCORE on Stanford RNA 3D Folding 2. I need you to act as an autonomous researcher. Use the web_fetch or google_web_search tools to find new architectures (like AlphaFold3, deep learning constraints, or MSA processing tricks). Create a new, improved python script for the kernel and save it as notebooks/script_v${NEXT_VER}.py using the write_file tool. DO NOT create a .ipynb file, just the raw python script. Make sure it still includes the offline installer logic at the top."
                    
                    break
                fi
            done
            break # Break inner loop
            
        elif [[ $STATUS == *"ERROR"* ]]; then
            echo "$(date) | ❌ ERROR! Downloading logs for AI debugging..." | tee -a $LOG_FILE
            mkdir -p notebooks/logs_v$VERSION
            kaggle kernels output $CURRENT_KERNEL -p notebooks/logs_v$VERSION --force
            
            NEXT_VER=$((VERSION + 1))
            echo "🧠 Invoking Gemini AI for Debugging and fixing v$NEXT_VER..." | tee -a $LOG_FILE
            
            # Call Gemini in headless mode to read logs and fix the code
            gemini -y -p "The Kaggle kernel $CURRENT_KERNEL failed. Read the error logs inside notebooks/logs_v${VERSION}/ using read_file or run_shell_command('tail'). Also read the current source code of the notebook (you can pull it with kaggle kernels pull $CURRENT_KERNEL -m). Fix the python code and save the entirely corrected python script as notebooks/script_v${NEXT_VER}.py using the write_file tool. DO NOT create a .ipynb file, just the raw python script."
            
            break # Break inner loop
        fi
        sleep 60
    done
    
    # After Gemini finishes, package the python script into a notebook and push
    if [ -f "notebooks/script_v${NEXT_VER}.py" ]; then
        echo "$(date) | 📦 Packaging v$NEXT_VER notebook..." | tee -a $LOG_FILE
        python3 -c "
import json
with open('notebooks/script_v${NEXT_VER}.py', 'r') as f: code = f.read()
nb = {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {'trusted': True}, 'outputs': [], 'source': [code]}], 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 4}
with open('notebooks/rna_folding_v${NEXT_VER}.ipynb', 'w') as f: json.dump(nb, f, indent=1)
"
        
        CURRENT_KERNEL="softkleenex/rna-folding-v${NEXT_VER}-auto"
        cat > notebooks/kernel-metadata.json <<EOF
{
  "id": "$CURRENT_KERNEL",
  "title": "RNA Folding v${NEXT_VER} Auto",
  "code_file": "rna_folding_v${NEXT_VER}.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "dataset_sources": [
    "qiweiyin/protenix-v1-adjusted",
    "yuriygreben/stanford-rna-protenix-wheels-3-12"
  ],
  "competition_sources": ["stanford-rna-3d-folding-2"]
}
EOF
        echo "$(date) | 🚀 Pushing v$NEXT_VER..." | tee -a $LOG_FILE
        kaggle kernels push -p notebooks
        sleep 30
        VERSION=$NEXT_VER
    else
        echo "❌ Gemini failed to produce notebooks/script_v${NEXT_VER}.py. Retrying AI step in 1 minute..." | tee -a $LOG_FILE
        sleep 60
    fi
done

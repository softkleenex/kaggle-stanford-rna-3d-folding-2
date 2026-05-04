#!/bin/bash
# INFINITE BROAD LOOP V5: The Architect
# This loop uses Gemini to analyze and improve the BEST script (v24/v27).

COMPETITION="stanford-rna-3d-folding-2"
LOG_FILE="infinite_broad.log"
HISTORY_FILE="history.log"
BEST_SCORE=0.432
VERSION=35

echo "=== STARTING INFINITE BROAD LOOP V5 ===" > $LOG_FILE

while true; do
    log_msg="--- Version v$VERSION START ---"
    echo "$(date) | $log_msg" | tee -a $LOG_FILE
    
    # 1. AI EVOLUTION: Patch the best script
    # We use notebooks/top1/script.py as the foundation
    echo "$(date) | 🧠 AI is designing improvements for v$VERSION..." | tee -a $LOG_FILE
    
    gemini -y -p "You are an expert bioinformatician. We have a base script in 'notebooks/top1/script.py' that scores 0.432. 
    Our recent attempts (v29-v34) failed to improve it because they simplified the logic too much.
    
    TASK: Improve 'notebooks/top1/script.py' by applying ONE high-level structural improvement.
    IDEAS:
    1. Improve 'adaptive_rna_constraints' by adding more physical realism (e.g., base-pairing constraints).
    2. Enhance 'find_similar_sequences_detailed' with secondary structure similarity.
    3. Add a 'refinement' pass using RDKit to fix bond lengths.
    
    RULES:
    1. Read 'notebooks/top1/script.py' first.
    2. APPLY the improvement directly to that code.
    3. Keep the offline wheel installer and path resolution logic.
    4. Save the ENTIRE improved python script to 'notebooks/script_v$VERSION.py'.
    
    BE CAREFUL: Do not lose the Kabsch alignment and chunk stitching logic. It is the core of our 0.432 score."

    if [ ! -f "notebooks/script_v$VERSION.py" ]; then
        echo "❌ AI failed to produce script. Retrying in 2 mins..." | tee -a $LOG_FILE
        sleep 120
        continue
    fi

    # 2. PACKAGE & PUSH
    echo "$(date) | 📦 Packaging v$VERSION..." | tee -a $LOG_FILE
    python3 -c "
import json
with open('notebooks/script_v$VERSION.py', 'r') as f: code = f.read()
nb = {'cells': [{'cell_type': 'code', 'metadata': {'trusted': True}, 'outputs': [], 'source': [code]}], 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 4}
with open('notebooks/rna_folding_v$VERSION.ipynb', 'w') as f: json.dump(nb, f, indent=1)
"
    
    KERNEL_ID="softkleenex/rna-folding-v$VERSION-auto"
    cat > notebooks/kernel-metadata.json <<META
{
  "id": "$KERNEL_ID",
  "title": "RNA Folding v$VERSION Auto",
  "code_file": "rna_folding_v$VERSION.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "dataset_sources": ["qiweiyin/protenix-v1-adjusted", "yuriygreben/stanford-rna-protenix-wheels-3-12"],
  "competition_sources": ["$COMPETITION"]
}
META

    echo "$(date) | 🚀 Pushing v$VERSION..." | tee -a $LOG_FILE
    kaggle kernels push -p notebooks
    
    # 3. MONITOR
    echo "$(date) | ⏳ Monitoring v$VERSION..." | tee -a $LOG_FILE
    while true; do
        STATUS=$(kaggle kernels status "$KERNEL_ID" 2>/dev/null)
        if [[ "$STATUS" == *"COMPLETE"* ]]; then
            echo "$(date) | ✅ COMPLETE!" | tee -a $LOG_FILE
            break
        elif [[ "$STATUS" == *"ERROR"* ]]; then
            echo "$(date) | ❌ FAILED! Logging error..." | tee -a $LOG_FILE
            mkdir -p "notebooks/logs_v$VERSION"
            kaggle kernels output "$KERNEL_ID" -p "notebooks/logs_v$VERSION" --force
            echo "$(date) | v$VERSION | FAILED" >> "$HISTORY_FILE"
            VERSION=$((VERSION + 1))
            continue 2
        fi
        sleep 180
    done
    
    # 4. SUBMIT
    echo "$(date) | 📤 Submitting..." | tee -a $LOG_FILE
    kaggle competitions submit -c "$COMPETITION" -k "$KERNEL_ID" -f submission.csv -m "Auto v$VERSION" -v 1
    
    # 5. WAIT FOR SCORE
    echo "$(date) | 🏆 Scoring..." | tee -a $LOG_FILE
    while true; do
        LATEST=$(kaggle competitions submissions -c "$COMPETITION" --csv | head -n 2 | tail -n 1)
        if [[ $LATEST == *"pending"* ]] || [[ $LATEST == *"scoring"* ]]; then
            sleep 60
        else
            SCORE=$(echo "$LATEST" | awk -F',' '{print $5}')
            echo "$(date) | 🎯 SCORE for v$VERSION: $SCORE" | tee -a $LOG_FILE
            echo "$(date) | v$VERSION | COMPLETE | Score: $SCORE" >> "$HISTORY_FILE"
            break
        fi
    done
    
    VERSION=$((VERSION + 1))
    sleep 60
done

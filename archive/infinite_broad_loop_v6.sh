#!/bin/bash
# INFINITE BROAD LOOP V6: Fully Automated, Single Kernel, Self-Learning
# 1. 완전 자동 (Fully automated loop)
# 2. 점유 커널은 단 하나 (Single Kernel updated via versioning)
# 3. 중복 시도 없음, 기록 제대로 (History tracking to prevent dupes)
# 4. 질의 고도화 (Dynamic prompting based on state)
# 5. 오류 및 디버깅 (Error log fed back into AI for debugging)
# 6. Fallback model: Try 3.1 Pro Preview first, then 3.1 Flash if rate-limited

COMPETITION="stanford-rna-3d-folding-2"
LOG_FILE="infinite_broad.log"
HISTORY_FILE="history.log"
KERNEL_ID="softkleenex/rna-folding-auto-pipeline"
PROMPT_FILE="current_prompt.txt"
BASE_SCRIPT="notebooks/top1/script.py"

echo "=== STARTING INFINITE BROAD LOOP V6 ===" >> $LOG_FILE

# Initialize version
if [ ! -f "current_version.txt" ]; then
    echo "36" > current_version.txt
fi

# Ensure history file exists
if [ ! -f "$HISTORY_FILE" ]; then
    echo "Version | Status | Score | Idea" > "$HISTORY_FILE"
fi

# We use a single kernel ID. Metadata is generated once unless it changes.
cat > notebooks/kernel-metadata.json <<META
{
  "id": "$KERNEL_ID",
  "title": "RNA Folding Auto Pipeline",
  "code_file": "rna_folding_auto.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "false",
  "dataset_sources": [
    "qiweiyin/protenix-v1-adjusted",
    "yuriygreben/stanford-rna-protenix-wheels-3-12",
    "kami1976/biopython-cp312",
    "amirrezaaleyasin/biotite",
    "amirrezaaleyasin/rdkit-2025-9-5"
  ],
  "competition_sources": ["$COMPETITION"]
}
META

LAST_STATE="NEW" # NEW, FAILED, SCORED

while true; do
    VERSION=$(cat current_version.txt)
    echo "--------------------------------------------------" | tee -a $LOG_FILE
    echo "$(date) | 🚀 STARTING v$VERSION | Last State: $LAST_STATE" | tee -a $LOG_FILE
    
    TARGET_SCRIPT="notebooks/script_v${VERSION}.py"
    
    # 1. CONSTRUCT ADVANCED PROMPT
    echo "$(date) | 🧠 Constructing Prompt..." | tee -a $LOG_FILE
    
    cat <<EOF > $PROMPT_FILE
You are an expert AI bioinformatician autonomously competing in a Kaggle competition.
Your base script is '$BASE_SCRIPT' (Score ~0.432).
Your goal is to write a highly optimized, fully complete python script and save it to '$TARGET_SCRIPT'.

Here is the history of previous attempts to avoid duplicates:
$(tail -n 20 $HISTORY_FILE)

EOF

    if [ "$LAST_STATE" == "FAILED" ]; then
        echo "The previous version (v$((VERSION-1))) FAILED with the following error log:" >> $PROMPT_FILE
        cat notebooks/logs_v$((VERSION-1))/*.log 2>/dev/null | tail -n 100 >> $PROMPT_FILE
        echo "" >> $PROMPT_FILE
        cat <<EOF >> $PROMPT_FILE
TASK:
1. DEBUG: Use the 'debugger' sub-agent tool to analyze the error log and identify the exact root cause in the python script.
2. FIX: Delegate to the 'coder' sub-agent tool to implement the fix into '$TARGET_SCRIPT'. Do not introduce new features, just fix the error.
3. SUMMARY: Write a brief 1-line summary of the bug and your fix to 'current_strategy.txt'.
EOF
    else
        cat <<EOF >> $PROMPT_FILE
TASK: You are the Orchestrator. You must delegate to your sub-agents to achieve the following:
1. RESEARCH: Call the 'researcher' sub-agent. Instruct it to use web search or Kaggle CLI to discover the latest top-scoring public solutions, discussions, or papers for 'stanford-rna-3d-folding-2'.
2. PLAN: Call the 'planner' sub-agent. Instruct it to review the research and the history to formulate ONE unique, high-impact structural or physical improvement. It must explicitly avoid repeating past failed/low-scoring ideas.
3. IMPLEMENT: Call the 'coder' sub-agent. Instruct it to read '$BASE_SCRIPT' and meticulously apply the planner's improvement to the code.

RULES:
1. Maintain the offline wheel installer, chunk stitching, and Kabsch alignment logic exactly.
2. The final valid python code MUST be saved to '$TARGET_SCRIPT'.
3. Write a brief 1-line summary of the strategy to 'current_strategy.txt'.
EOF
    fi

    # 2. RUN AI WITH MODEL FALLBACK
    echo "$(date) | 🤖 Gemini is thinking (Trying 3.1 Pro Preview)..." | tee -a $LOG_FILE
    
    # Attempt with Pro first using cat to pass file contents correctly
    gemini -y -m gemini-3.1-pro-preview -p "$(cat $PROMPT_FILE)"
    
    # Check if target script was created (success metric)
    if [ ! -f "$TARGET_SCRIPT" ]; then
        echo "$(date) | ⚠️ Pro failed or hit rate limit. Trying 3.1 Flash..." | tee -a $LOG_FILE
        gemini -y -m gemini-3.1-flash -p "$(cat $PROMPT_FILE)"
    fi
    
    if [ ! -f "$TARGET_SCRIPT" ]; then
        echo "❌ AI failed to produce $TARGET_SCRIPT. Retrying in 60s..." | tee -a $LOG_FILE
        sleep 60
        continue
    fi
    
    STRATEGY="Unknown Strategy"
    if [ -f "current_strategy.txt" ]; then
        STRATEGY=$(cat current_strategy.txt)
    fi

    # 3. PACKAGE TO IPYNB
    echo "$(date) | 📦 Packaging v$VERSION..." | tee -a $LOG_FILE
    python3 -c "
import json
with open('$TARGET_SCRIPT', 'r') as f: code = f.read()
nb = {'cells': [{'cell_type': 'code', 'metadata': {'trusted': True}, 'outputs': [], 'source': [code]}], 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 4}
with open('notebooks/rna_folding_auto.ipynb', 'w') as f: json.dump(nb, f, indent=1)
"
    
    # 4. PUSH KERNEL
    echo "$(date) | 🚀 Pushing to Kernel ($KERNEL_ID)..." | tee -a $LOG_FILE
    kaggle kernels push -p notebooks
    
    # 5. MONITOR KERNEL
    echo "$(date) | ⏳ Monitoring Kernel Execution..." | tee -a $LOG_FILE
    while true; do
        STATUS=$(kaggle kernels status "$KERNEL_ID" 2>/dev/null)
        if [[ "$STATUS" == *"COMPLETE"* ]]; then
            echo "$(date) | ✅ Kernel COMPLETE!" | tee -a $LOG_FILE
            LAST_STATE="COMPLETE"
            break
        elif [[ "$STATUS" == *"ERROR"* ]]; then
            echo "$(date) | ❌ Kernel FAILED!" | tee -a $LOG_FILE
            mkdir -p "notebooks/logs_v$VERSION"
            kaggle kernels output "$KERNEL_ID" -p "notebooks/logs_v$VERSION" --force
            echo "v$VERSION | FAILED | N/A | $STRATEGY" >> "$HISTORY_FILE"
            LAST_STATE="FAILED"
            break
        fi
        sleep 60
    done
    
    if [ "$LAST_STATE" == "FAILED" ]; then
        VERSION=$((VERSION + 1))
        echo "$VERSION" > current_version.txt
        continue
    fi
    
    # 6. SUBMIT TO COMPETITION
    echo "$(date) | 📤 Submitting to Kaggle..." | tee -a $LOG_FILE
    kaggle competitions submit -c "$COMPETITION" -k "$KERNEL_ID" -f submission.csv -m "Auto v$VERSION: $STRATEGY" -v 1
    
    # 7. WAIT FOR SCORE
    echo "$(date) | 🏆 Waiting for Score..." | tee -a $LOG_FILE
    while true; do
        LATEST=$(kaggle competitions submissions -c "$COMPETITION" --csv | head -n 2 | tail -n 1)
        STATUS=$(echo "$LATEST" | awk -F',' '{print $4}')
        
        if [[ "$STATUS" == "pending" ]] || [[ "$STATUS" == "scoring" ]] || [[ "$STATUS" == "None" ]]; then
            sleep 60
        elif [[ "$STATUS" == "error" ]]; then
             echo "$(date) | ❌ Submission Error!" | tee -a $LOG_FILE
             echo "v$VERSION | SUB_ERROR | N/A | $STRATEGY" >> "$HISTORY_FILE"
             LAST_STATE="FAILED"
             break
        else
            SCORE=$(echo "$LATEST" | awk -F',' '{print $5}')
            echo "$(date) | 🎯 SCORE for v$VERSION: $SCORE" | tee -a $LOG_FILE
            echo "v$VERSION | SCORED | $SCORE | $STRATEGY" >> "$HISTORY_FILE"
            LAST_STATE="SCORED"
            break
        fi
    done
    
    VERSION=$((VERSION + 1))
    echo "$VERSION" > current_version.txt
    sleep 30
done
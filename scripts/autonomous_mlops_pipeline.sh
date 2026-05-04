#!/bin/bash
# INFINITE BROAD LOOP V9: The Self-Aware Autonomous Grandmaster (Local Validation Edition)
# Key Features:
# 1. Dynamic Knowledge Harvesting (Scout)
# 2. Meta-Learning Memory (Reflector)
# 3. Local QA Pre-flight Check (Tester)
# 4. Multi-Agent Orchestration & Single Kernel
# 5. Offline Research Mode (Graceful GPU Quota Handling)
# 6. Kaggle Local Validation Mode (Graceful Submission Limit Handling)

COMPETITION="stanford-rna-3d-folding-2"
LOG_FILE="infinite_broad_v9.log"
HISTORY_FILE="history_v9.log"
LEARNINGS_FILE="learnings.md"
KERNEL_ID="softkleenex/rna-folding-auto-pipeline"
PROMPT_FILE="current_prompt.txt"
BASE_SCRIPT="notebooks/top1/script.py"

echo "=== STARTING INFINITE BROAD LOOP V9 ===" >> $LOG_FILE

if [ ! -f "current_version.txt" ]; then
    echo "38" > current_version.txt
fi

if [ ! -f "$HISTORY_FILE" ]; then
    # Copy from previous history if exists
    if [ -f "history_v8.log" ]; then
        cp history_v8.log $HISTORY_FILE
    else
        echo "Version | Status | Score | Idea" > "$HISTORY_FILE"
    fi
fi

if [ ! -f "$LEARNINGS_FILE" ]; then
    echo "# Permanent Learnings" > "$LEARNINGS_FILE"
    echo "- Always ensure strict python syntax." >> "$LEARNINGS_FILE"
fi

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

LAST_STATE="NEW"
GPU_QUOTA_EXHAUSTED=false
SUBMISSION_EXHAUSTED=false

while true; do
    # Check for Quota Reset (Kaggle resets daily for submissions, Weekly for GPU)
    CURRENT_UTC_HOUR=$(date -u +%H)
    CURRENT_UTC_DAY=$(date -u +%a) # e.g., Sun, Mon...
    
    # 1. Submission Daily Reset
    if [ "$CURRENT_UTC_HOUR" == "00" ] || [ "$CURRENT_UTC_HOUR" == "01" ]; then
        if [ "$SUBMISSION_EXHAUSTED" = true ]; then
            echo "$(date) | 🌅 New Day (UTC)! Resetting Submission Quota Flag." | tee -a $LOG_FILE
            SUBMISSION_EXHAUSTED=false
        fi
    fi
    
    # 2. GPU Weekly Reset (Sunday 00:00 UTC is ~09:00 KST)
    if [ "$CURRENT_UTC_DAY" == "Sun" ] && ([ "$CURRENT_UTC_HOUR" == "00" ] || [ "$CURRENT_UTC_HOUR" == "01" ]); then
        if [ "$GPU_QUOTA_EXHAUSTED" = true ]; then
            echo "$(date) | 📅 Sunday Reset (UTC)! Clearing GPU Quota Exhausted Flag." | tee -a $LOG_FILE
            GPU_QUOTA_EXHAUSTED=false
        fi
    fi

    VERSION=$(cat current_version.txt)
    echo "--------------------------------------------------" | tee -a $LOG_FILE
    echo "$(date) | 🚀 STARTING v$VERSION | Last State: $LAST_STATE" | tee -a $LOG_FILE
    echo "GPU Quota Exhausted: $GPU_QUOTA_EXHAUSTED | Submissions Exhausted: $SUBMISSION_EXHAUSTED" | tee -a $LOG_FILE
    
    TARGET_SCRIPT="notebooks/script_v${VERSION}.py"
    
    # ---------------------------------------------------------
    # PHASE 0: REFLECTION (Meta-Learning)
    # ---------------------------------------------------------
    if [ "$LAST_STATE" == "SCORED" ] || [[ "$LAST_STATE" == "LOCAL_SCORED"* ]]; then
        echo "$(date) | 🪞 Reflecting on previous score..." | tee -a $LOG_FILE
        cat <<EOF > reflect_prompt.txt
We just received a new score (could be Kaggle LB or Local Validation). Here is our history:
$(tail -n 10 $HISTORY_FILE)

Based on the score trajectory, did the last strategy work or fail? 
Extract ONE concise, permanent technical lesson from this attempt (e.g., "Do not use RMSD scaling for X" or "Increasing chunk overlap improves score"). 
Keep it under 2 sentences.
EOF
        LESSON=$(gemini -y -m gemini-3.1-flash -p "$(cat reflect_prompt.txt)" 2>/dev/null)
        echo "- v$((VERSION-1)): $LESSON" >> "$LEARNINGS_FILE"
    fi

    # ---------------------------------------------------------
    # PHASE 1: KNOWLEDGE HARVESTING (Scout)
    # ---------------------------------------------------------
    if [ "$LAST_STATE" != "FAILED_LOCAL" ] && [ "$LAST_STATE" != "FAILED_KAGGLE" ]; then
        echo "$(date) | 🕵️ Scouting Kaggle for new context..." | tee -a $LOG_FILE
        SORT_OPTS=("voteCount" "scoreDesc" "hotness" "dateCreated")
        RAND_INDEX=$((RANDOM % 4))
        SORT_BY=${SORT_OPTS[$RAND_INDEX]}
        
        echo "  - Querying Top Notebooks by: $SORT_BY" | tee -a $LOG_FILE
        kaggle kernels list --competition "$COMPETITION" --sort-by "$SORT_BY" -p 1 --csv > kernels.csv
        TARGET_KERNEL=$(awk -F',' 'NR==2 {print $1}' kernels.csv | tr -d '"')
        
        echo "  - Pulling code from $TARGET_KERNEL" | tee -a $LOG_FILE
        mkdir -p tmp_kernel
        kaggle kernels pull "$TARGET_KERNEL" -p tmp_kernel 2>/dev/null
        
        IPYNB_FILE=$(ls tmp_kernel/*.ipynb 2>/dev/null | head -n 1)
        if [ -n "$IPYNB_FILE" ]; then
            python3 tools/parse_ipynb.py "$IPYNB_FILE" competitor_context.txt
        else
            echo "No ipynb found in kernel download." > competitor_context.txt
        fi
        rm -rf tmp_kernel kernels.csv
    fi

    # ---------------------------------------------------------
    # PHASE 2: CONSTRUCT PROMPT & GENERATE CODE
    # ---------------------------------------------------------
    echo "$(date) | 🧠 Constructing Prompt..." | tee -a $LOG_FILE
    
    cat <<EOF > $PROMPT_FILE
You are an expert AI bioinformatician autonomously competing in the 'stanford-rna-3d-folding-2' Kaggle competition.
Your base script is '$BASE_SCRIPT'.
Your goal is to write a highly optimized, fully complete python script and save it to '$TARGET_SCRIPT'.

### PERMANENT LEARNINGS (DO NOT VIOLATE THESE)
$(cat $LEARNINGS_FILE)

### RECENT HISTORY
$(tail -n 15 $HISTORY_FILE)

### CRITICAL REQUIREMENT FOR LOCAL VALIDATION (IMPORTANT)
Regardless of the strategy, you MUST implement a local validation split from \`train.csv\`. 
Evaluate your model's RMSD on this validation set during the script execution.
You MUST print the final validation score to stdout exactly in this format:
\`LOCAL_VALIDATION_RMSD: 0.12345\`
This is required because the automated pipeline relies on parsing this string from the logs if Kaggle submission limits are reached.
EOF

    if [ "$LAST_STATE" == "FAILED_KAGGLE" ]; then
        echo "### KAGGLE RUNTIME ERROR" >> $PROMPT_FILE
        echo "The previous Kaggle run failed with:" >> $PROMPT_FILE
        cat notebooks/logs_v$((VERSION-1))/*.log 2>/dev/null | tail -n 100 >> $PROMPT_FILE
        echo "" >> $PROMPT_FILE
        cat <<EOF >> $PROMPT_FILE
TASK:
1. DEBUG: Analyze the error log and identify the exact root cause in the python script.
2. FIX: Implement the fix into '$TARGET_SCRIPT'. Do not introduce new features, just fix the bug.
3. SUMMARY: Write a brief 1-line summary of your fix to 'current_strategy.txt'.
EOF
    elif [ "$LAST_STATE" == "FAILED_LOCAL" ]; then
        echo "### LOCAL SYNTAX ERROR" >> $PROMPT_FILE
        echo "The script failed local py_compile with:" >> $PROMPT_FILE
        cat local_error.log >> $PROMPT_FILE
        echo "" >> $PROMPT_FILE
        cat <<EOF >> $PROMPT_FILE
TASK:
1. FIX SYNTAX: You made a critical syntax or Indentation error. Fix it in '$TARGET_SCRIPT'. Do not change the logic.
2. SUMMARY: Write a brief 1-line summary of the syntax fix to 'current_strategy.txt'.
EOF
    else
        echo "### KAGGLE INTELLIGENCE (Recently Scraped Competitor Code Snippet)" >> $PROMPT_FILE
        cat competitor_context.txt >> $PROMPT_FILE
        echo "" >> $PROMPT_FILE
        if [ "$GPU_QUOTA_EXHAUSTED" = true ]; then
            echo "NOTE: We are currently in PURE OFFLINE RESEARCH MODE because GPU API limits were reached." >> $PROMPT_FILE
            echo "Please generate a high-quality experimental variation. It will be validated locally and stockpiled for later." >> $PROMPT_FILE
        elif [ "$SUBMISSION_EXHAUSTED" = true ]; then
            echo "NOTE: We are in KAGGLE KERNEL LOCAL VALIDATION MODE because daily submissions are exhausted." >> $PROMPT_FILE
            echo "We will run the kernel on Kaggle but rely entirely on your printed LOCAL_VALIDATION_RMSD for evaluation. Make sure it is accurate." >> $PROMPT_FILE
        fi
        cat <<EOF >> $PROMPT_FILE
TASK: You are the Orchestrator. Delegate to your sub-agents:
1. PLAN: Review our History, Learnings, and the Competitor Context. Formulate ONE unique, high-impact structural or physical improvement.
2. IMPLEMENT: Read '$BASE_SCRIPT' and meticulously apply your improvement to the code.
3. RULES:
   - Maintain the offline wheel installer, chunk stitching, and Kabsch alignment logic exactly.
   - Ensure the code is 100% syntactically valid Python 3.12.
   - Save the valid python code to '$TARGET_SCRIPT'.
   - Write a brief 1-line summary of the strategy to 'current_strategy.txt'.
EOF
    fi

    # ---------------------------------------------------------
    # PHASE 2: TWO-STAGE AI GENERATION
    # ---------------------------------------------------------
    # STAGE 2A: STRATEGIC PLANNING (Using Flash)
    echo "$(date) | 🧠 Phase 2A: Strategic Planning (3-Flash)..." | tee -a $LOG_FILE
    PLAN_PROMPT_FILE="tmp_plan_prompt.txt"
    cat <<EOF > $PLAN_PROMPT_FILE
You are a Senior Structural Biologist. Review the following context:
1. HISTORY: $(tail -n 5 $HISTORY_FILE)
2. COMPETITOR CONTEXT: $(cat $COMPETITOR_CONTEXT_FILE | head -n 20)
3. CURRENT PLAN: $(cat current_plan.md)

TASK: Suggest ONE specific, physically-grounded improvement for our RNA folding script.
OUTPUT: A brief 1-line strategy summary only.
EOF
    
    export GEMINI_API_KEY=$(head -n 1 api_keys.txt)
    STRATEGY=$(gemini -y -m gemini-3-flash-preview -p "$(cat $PLAN_PROMPT_FILE)" 2>&1 | tail -n 1)
    echo "STRATEGY: $STRATEGY" > current_strategy.txt
    echo "$(date) | 💡 Strategic Plan: $STRATEGY" | tee -a $LOG_FILE

    # STAGE 2B: CODE IMPLEMENTATION (Using Pro)
    echo "$(date) | 🤖 Phase 2B: Code Implementation (3.1-Pro)... [Key ending in ${GEMINI_API_KEY: -4}]" | tee -a $LOG_FILE
    
    cat <<EOF > $PROMPT_FILE
TASK: Implement the following strategy into our RNA folding script: "$STRATEGY"
BASE SCRIPT: $(cat $BASE_SCRIPT | head -n 50)... (rest of code)
RULES:
1. Maintain all offline installers and chunking logic.
2. CRITICAL: Use dynamic path resolution (e.g., Path('/kaggle/input').rglob('*.csv')) to find datasets. NEVER hardcode paths like '/kaggle/input/stanford-rna-3d-folding-2/'.
3. Ensure 100% syntactically valid Python 3.12.
4. Save the full code to '$TARGET_SCRIPT'.
EOF

    GEMINI_OUT=$(gemini -y -m gemini-3.1-pro-preview -p "$(cat $PROMPT_FILE)" 2>&1)
    
    # Check for Quota or 429 Errors
    if [ ! -f "$TARGET_SCRIPT" ] || echo "$GEMINI_OUT" | grep -qiE "429|quota|exhausted"; then
        if echo "$GEMINI_OUT" | grep -qiE "429|quota|exhausted"; then
            echo "$(date) | 🔄 Pro Quota hit! Rotating Key..." | tee -a $LOG_FILE
            tail -n +2 api_keys.txt > temp_keys.txt
            head -n 1 api_keys.txt >> temp_keys.txt
            mv temp_keys.txt api_keys.txt
            export GEMINI_API_KEY=$(head -n 1 api_keys.txt)
            echo "$(date) | 🔄 Retrying Pro with Key ending in ${GEMINI_API_KEY: -4}..." | tee -a $LOG_FILE
            GEMINI_OUT=$(gemini -y -m gemini-3.1-pro-preview -p "$(cat $PROMPT_FILE)" 2>&1)
        fi
        
        if [ ! -f "$TARGET_SCRIPT" ]; then
            echo "$(date) | ⚠️ Pro failed. Falling back to 3-Flash for implementation..." | tee -a $LOG_FILE
            gemini -y -m gemini-3-flash-preview -p "$(cat $PROMPT_FILE)" > /dev/null 2>&1
        fi
    fi
    
    if [ ! -f "$TARGET_SCRIPT" ]; then
        echo "❌ AI failed to produce $TARGET_SCRIPT. Retrying in 60s..." | tee -a $LOG_FILE
        sleep 60
        continue
    fi

    # ---------------------------------------------------------
    # PHASE 3: LOCAL QA (Pre-flight syntax check)
    # ---------------------------------------------------------
    echo "$(date) | 🔍 Local QA: Compiling $TARGET_SCRIPT..." | tee -a $LOG_FILE
    if ! python3 -m py_compile "$TARGET_SCRIPT" 2> local_error.log; then
        echo "$(date) | ❌ LOCAL SYNTAX ERROR! Taking a 10-minute break to prevent infinite loops..." | tee -a $LOG_FILE
        LAST_STATE="FAILED_LOCAL"
        rm -f "$TARGET_SCRIPT" # Remove corrupted file
        sleep 600
        continue
    fi
    echo "$(date) | ✅ Local QA Passed!" | tee -a $LOG_FILE

    STRATEGY="Unknown Strategy"
    if [ -f "current_strategy.txt" ]; then
        STRATEGY=$(cat current_strategy.txt)
    fi

    # ---------------------------------------------------------
    # OFFLINE RESEARCH MODE (GPU Quota Exhausted)
    # ---------------------------------------------------------
    # PHASE 4: PACKAGE & DEPLOY
    # ---------------------------------------------------------
    echo "$(date) | 📦 Packaging v$VERSION..." | tee -a $LOG_FILE
    python3 -c "
import json
with open('$TARGET_SCRIPT', 'r') as f: code = f.read()
nb = {'cells': [{'cell_type': 'code', 'metadata': {'trusted': True}, 'outputs': [], 'source': [code]}], 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 4}
with open('notebooks/rna_folding_auto.ipynb', 'w') as f: json.dump(nb, f, indent=1)
"
    
    echo "$(date) | 🚀 Pushing to Kernel ($KERNEL_ID)..." | tee -a $LOG_FILE
    PUSH_OUT=$(kaggle kernels push -p notebooks 2>&1)
    echo "$PUSH_OUT" | tee -a $LOG_FILE

    if echo "$PUSH_OUT" | grep -qiE "quota|limit"; then
        echo "$(date) | 🚨 GPU Quota Exceeded Detected during push!" | tee -a $LOG_FILE
        GPU_QUOTA_EXHAUSTED=true
        
        echo "$(date) | 🔄 Switching kernel to CPU mode and retrying push..." | tee -a $LOG_FILE
        sed -i '' 's/"enable_gpu": "true"/"enable_gpu": "false"/g' notebooks/kernel-metadata.json
        
        PUSH_OUT_CPU=$(kaggle kernels push -p notebooks 2>&1)
        echo "$PUSH_OUT_CPU" | tee -a $LOG_FILE
        
        if echo "$PUSH_OUT_CPU" | grep -qiE "quota|limit|error"; then
            echo "$(date) | 🛑 CPU Push also failed or quota exhausted. Saving as Candidate." | tee -a $LOG_FILE
            mkdir -p candidates
            cp "$TARGET_SCRIPT" "candidates/candidate_v${VERSION}.py"
            echo "v$VERSION | CANDIDATE | N/A | $STRATEGY" >> "$HISTORY_FILE"
            LAST_STATE="CANDIDATE"
            
            # Since we can't push, pace the loop with a 5 minute sleep instead of 30 mins to keep generating
            echo "$(date) | 💤 Pacing API for 5 minutes..." | tee -a $LOG_FILE
            sleep 300
            
            VERSION=$((VERSION + 1))
            echo "$VERSION" > current_version.txt
            
            # Reset kernel metadata back to GPU for next iteration in case quota resets
            sed -i '' 's/"enable_gpu": "false"/"enable_gpu": "true"/g' notebooks/kernel-metadata.json
            continue
        else
            echo "$(date) | ✅ Pushed successfully as CPU kernel!" | tee -a $LOG_FILE
            # Let it fall through to Phase 5 (Monitoring)
        fi
    else
        # SUCCESSFUL PUSH! (Means Quota is back)
        if [ "$GPU_QUOTA_EXHAUSTED" = true ]; then
            echo "$(date) | 🎉 SUCCESS! GPU Quota is back. Resuming SUBMISSION mode." | tee -a $LOG_FILE
            GPU_QUOTA_EXHAUSTED=false
        fi
    fi
    
    # ---------------------------------------------------------
    # PHASE 5: MONITORING
    # ---------------------------------------------------------
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
            LAST_STATE="FAILED_KAGGLE"
            break
        fi
        sleep 60
    done
    
    if [ "$LAST_STATE" == "FAILED_KAGGLE" ]; then
        VERSION=$((VERSION + 1))
        echo "$VERSION" > current_version.txt
        continue
    fi
    
    # ---------------------------------------------------------
    # LOCAL VALIDATION MODE (If Submissions Exhausted)
    # ---------------------------------------------------------
    if [ "$SUBMISSION_EXHAUSTED" = true ]; then
        echo "$(date) | 🕵️‍♂️ Submissions Exhausted. Extracting LOCAL_VALIDATION_RMSD from kernel logs..." | tee -a $LOG_FILE
        mkdir -p "notebooks/logs_v$VERSION"
        kaggle kernels output "$KERNEL_ID" -p "notebooks/logs_v$VERSION" --force >/dev/null 2>&1
        
        # Look for LOCAL_VALIDATION_RMSD in downloaded logs
        LOCAL_SCORE=$(grep -h "LOCAL_VALIDATION_RMSD:" notebooks/logs_v$VERSION/*.log 2>/dev/null | awk '{print $2}' | tail -n 1)
        
        if [ -n "$LOCAL_SCORE" ]; then
            echo "$(date) | 🎯 LOCAL SCORE for v$VERSION: $LOCAL_SCORE" | tee -a $LOG_FILE
            echo "v$VERSION | LOCAL_SCORED | $LOCAL_SCORE | $STRATEGY" >> "$HISTORY_FILE"
            LAST_STATE="LOCAL_SCORED_$LOCAL_SCORE"
        else
            echo "$(date) | ⚠️ Could not find LOCAL_VALIDATION_RMSD in logs." | tee -a $LOG_FILE
            echo "v$VERSION | LOCAL_FAILED | N/A | $STRATEGY" >> "$HISTORY_FILE"
            LAST_STATE="FAILED_LOCAL" # Force retry
        fi
        
        VERSION=$((VERSION + 1))
        echo "$VERSION" > current_version.txt
        sleep 30
        continue
    fi

    # ---------------------------------------------------------
    # PHASE 6: SCORING (Normal Submission Mode)
    # ---------------------------------------------------------
    echo "$(date) | 📤 Submitting to Kaggle..." | tee -a $LOG_FILE
    SUB_OUT=$(kaggle competitions submit -c "$COMPETITION" -k "$KERNEL_ID" -f submission.csv -m "Auto v$VERSION: $STRATEGY" -v 1 2>&1)
    echo "$SUB_OUT" | tee -a $LOG_FILE

    if echo "$SUB_OUT" | grep -qiE "limit|could not submit|400 client error|bad request"; then
        echo "$(date) | 🚨 Daily Submission Limit Reached Detected during submit!" | tee -a $LOG_FILE
        SUBMISSION_EXHAUSTED=true
        
        # We couldn't submit, but the kernel ran! Let's get the local score right now!
        echo "$(date) | 🕵️‍♂️ Falling back to Local Validation Score..." | tee -a $LOG_FILE
        mkdir -p "notebooks/logs_v$VERSION"
        kaggle kernels output "$KERNEL_ID" -p "notebooks/logs_v$VERSION" --force >/dev/null 2>&1
        LOCAL_SCORE=$(grep -h "LOCAL_VALIDATION_RMSD:" notebooks/logs_v$VERSION/*.log 2>/dev/null | awk '{print $2}' | tail -n 1)
        
        if [ -n "$LOCAL_SCORE" ]; then
            echo "$(date) | 🎯 LOCAL SCORE for v$VERSION: $LOCAL_SCORE" | tee -a $LOG_FILE
            echo "v$VERSION | LOCAL_SCORED | $LOCAL_SCORE | $STRATEGY" >> "$HISTORY_FILE"
            LAST_STATE="LOCAL_SCORED_$LOCAL_SCORE"
        else
            echo "$(date) | ⚠️ Could not find LOCAL_VALIDATION_RMSD in logs. Just treating as candidate." | tee -a $LOG_FILE
            echo "v$VERSION | CANDIDATE | N/A | $STRATEGY" >> "$HISTORY_FILE"
            LAST_STATE="CANDIDATE"
        fi
        
        VERSION=$((VERSION + 1))
        echo "$VERSION" > current_version.txt
        continue
    fi
    
    echo "$(date) | 🏆 Waiting for Score..." | tee -a $LOG_FILE
    while true; do
        LATEST=$(kaggle competitions submissions -c "$COMPETITION" --csv | head -n 2 | tail -n 1)
        STATUS=$(echo "$LATEST" | awk -F',' '{print $4}')
        
        if [[ "$STATUS" == "pending" ]] || [[ "$STATUS" == "scoring" ]] || [[ "$STATUS" == "None" ]]; then
            sleep 60
        elif [[ "$STATUS" == "error" ]]; then
             echo "$(date) | ❌ Submission Error!" | tee -a $LOG_FILE
             echo "v$VERSION | SUB_ERROR | N/A | $STRATEGY" >> "$HISTORY_FILE"
             LAST_STATE="FAILED_KAGGLE"
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

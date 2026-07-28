# Frontmatter Fix Report

Generated: 2026-06-05  
Vault: `/Users/turcinv/Documents/personal_knowledge/Career Knowledge Base`  
Files scanned: 1,639

---

# Summary

| Category | Count | Action taken |
|---|---|---|
| Valid YAML frontmatter | 1,611 | None needed |
| Auto-fixed (`heading_in_frontmatter`) | 1 | Fixed before scan ran |
| No frontmatter | 10 | 2 expected meta-files; 8 flagged for manual review |
| Unclosed frontmatter | 0 | — |
| YAML parse error | 18 | Flagged for manual review (unfilled `{{DATE}}` placeholder) |
| Missing required fields | 0 | — |
| Domain mismatch | 0 | — |

**Total files with issues: 29** (1 auto-fixed before scan, 28 flagged for manual review)

---

# Files Fixed

## `Templates/Knowledge Extraction Prompt - Fallback.md`

This file was fixed manually before the bulk scan ran (it was the trigger file). Backup: `Templates/Knowledge Extraction Prompt - Fallback.md.bak`.

**Issue 1 — Malformed title field (heading prefix in frontmatter)**

```diff
- ## title: Knowledge Extraction Prompt - Fallback
+ title: Knowledge Extraction Prompt - Fallback
```

**Issue 2 — Missing closing `---` before Markdown body**

The frontmatter block opened at line 1 but had no closing delimiter. The YAML parser consumed the entire Markdown table and heading (`# When to use which prompt`) as YAML content, then choked. Fix: inserted `---` after `status: processed` and before the first Markdown heading.

```diff
  status: processed
+ ---
+
  # When to use which prompt
```

YAML parse verified with `yaml.safe_load()` after fix — all 9 required fields parse correctly.

---

# Files Needing Manual Review

## Category 1 — YAML Parse Error: unfilled Templater `{{DATE}}` placeholder (18 files)

**Root cause:** These notes were created from an Obsidian Templater template but the `{{DATE}}` variable was never substituted. The double curly braces are parsed by YAML as a flow mapping, causing a parse error: `found unhashable key`.

**Fix required:** Replace `created: {{DATE}}` (and `updated: {{DATE}}` if present) with the actual ISO date (`YYYY-MM-DD`). Use `git log --follow --diff-filter=A -- <file>` or the file's modification date as a proxy.

**Example fix:**
```diff
- created: {{DATE}}
+ created: 2026-05-20
- updated: {{DATE}}
+ updated: 2026-06-05
```

**Affected files (18):**

| File | Domain |
|---|---|
| `Knowledge/AI & Automation/Gemini SDK Tools Argument Compatibility.md` | AI & Automation |
| `Knowledge/AI & Automation/Vertex AI RAG Request Structure.md` | AI & Automation |
| `Knowledge/Personal Knowledge/Linux Login and Boot Troubleshooting Timeline.md` | Personal Knowledge |
| `Knowledge/Software Engineering/Alarm SMS Message Construction.md` | Software Engineering |
| `Knowledge/Software Engineering/Diagnosing Fish Shell Login Problems.md` | Software Engineering |
| `Knowledge/Software Engineering/Handling PDF Metadata During Database Merges.md` | Software Engineering |
| `Knowledge/Software Engineering/IDE Type Checking Can Lag Behind Runtime SDK Behavior.md` | Software Engineering |
| `Knowledge/Software Engineering/JWT Alarm Activation Links.md` | Software Engineering |
| `Knowledge/Software Engineering/Professional Naming Conventions for SQLite Databases.md` | Software Engineering |
| `Knowledge/Software Engineering/Python Return Type Hints Should Match Runtime Behavior.md` | Software Engineering |
| `Knowledge/Software Engineering/SQLite Database Merge Strategy for FDA Submission Data.md` | Software Engineering |
| `Knowledge/Software Engineering/SQLite Idempotent Migration Patterns.md` | Software Engineering |
| `Knowledge/Software Engineering/SQLite Schema Standardization Workflow.md` | Software Engineering |
| `Knowledge/Software Engineering/SSH Access Failure Due to Firewall Rules.md` | Software Engineering |
| `Knowledge/Software Engineering/Secure Handling of Secrets in Python Services.md` | Software Engineering |
| `Knowledge/Software Engineering/Twilio SMS Service Design.md` | Software Engineering |
| `Knowledge/Software Engineering/USB Device Authorization Blocking Keyboard Input.md` | Software Engineering |
| `Knowledge/Software Engineering/Using dmesg to Distinguish Hardware and Shell Problems.md` | Software Engineering |

**Bulk fix command** (uses file modification time as the date proxy — verify before running):

```bash
cd "/Users/turcinv/Documents/personal_knowledge/Career Knowledge Base"
for f in \
  "Knowledge/AI & Automation/Gemini SDK Tools Argument Compatibility.md" \
  "Knowledge/AI & Automation/Vertex AI RAG Request Structure.md" \
  "Knowledge/Personal Knowledge/Linux Login and Boot Troubleshooting Timeline.md" \
  "Knowledge/Software Engineering/Alarm SMS Message Construction.md" \
  "Knowledge/Software Engineering/Diagnosing Fish Shell Login Problems.md" \
  "Knowledge/Software Engineering/Handling PDF Metadata During Database Merges.md" \
  "Knowledge/Software Engineering/IDE Type Checking Can Lag Behind Runtime SDK Behavior.md" \
  "Knowledge/Software Engineering/JWT Alarm Activation Links.md" \
  "Knowledge/Software Engineering/Professional Naming Conventions for SQLite Databases.md" \
  "Knowledge/Software Engineering/Python Return Type Hints Should Match Runtime Behavior.md" \
  "Knowledge/Software Engineering/SQLite Database Merge Strategy for FDA Submission Data.md" \
  "Knowledge/Software Engineering/SQLite Idempotent Migration Patterns.md" \
  "Knowledge/Software Engineering/SQLite Schema Standardization Workflow.md" \
  "Knowledge/Software Engineering/SSH Access Failure Due to Firewall Rules.md" \
  "Knowledge/Software Engineering/Secure Handling of Secrets in Python Services.md" \
  "Knowledge/Software Engineering/Twilio SMS Service Design.md" \
  "Knowledge/Software Engineering/USB Device Authorization Blocking Keyboard Input.md" \
  "Knowledge/Software Engineering/Using dmesg to Distinguish Hardware and Shell Problems.md"
do
  FILE_DATE=$(date -r "$f" +%Y-%m-%d)
  cp "$f" "$f.bak"
  sed -i '' "s/created: {{DATE}}/created: $FILE_DATE/" "$f"
  sed -i '' "s/updated: {{DATE}}/updated: 2026-06-05/" "$f"
  echo "Fixed: $f (date: $FILE_DATE)"
done
```

---

## Category 2 — No frontmatter: knowledge notes (8 files)

**Root cause:** These Jetson-related notes were added without YAML frontmatter. The RAG indexer will either skip them or index them without metadata.

**Fix required:** Add a complete frontmatter block at the top of each file. All belong to domain `Software Engineering`.

**Template to add at the top of each file:**
```yaml
---
title: <Note Title matching filename without .md>
domain: Software Engineering
type: Knowledge
created: <YYYY-MM-DD>
updated: 2026-06-05
tags:
  - jetson
  - linux
source: ChatGPT
confidence: high
status: processed
---
```

**Affected files (8):**

| File |
|---|
| `Knowledge/Software Engineering/Jetson Command-Only Troubleshooting Style.md` |
| `Knowledge/Software Engineering/Jetson Orin Nano Super Benchmarking Runbook.md` |
| `Knowledge/Software Engineering/Jetson Orin Nano Super CPU Benchmarking.md` |
| `Knowledge/Software Engineering/Jetson Orin Nano Super Clock and Power Tuning.md` |
| `Knowledge/Software Engineering/Jetson Orin Nano Super Sensors and Thermal Zones.md` |
| `Knowledge/Software Engineering/Jetson Orin Nano Super jtop Limitations.md` |
| `Knowledge/Software Engineering/Jetson PyTorch on JetPack 6.md` |
| `Knowledge/Software Engineering/OpenCV CUDA on JetPack 6.md` |

---

## Category 3 — No frontmatter: expected meta-files (2 files, no action needed)

These files intentionally have no YAML frontmatter — they are vault-level configuration/documentation files, not knowledge notes.

| File | Reason |
|---|---|
| `CLAUDE.md` | AI assistant guidance — not a knowledge note |
| `README.md` | Vault/repo documentation — not a knowledge note |

---

# Common Frontmatter Issues Found

## 1. Unfilled Templater placeholder `{{DATE}}`

- **Frequency:** 18 files
- **Pattern:** `created: {{DATE}}` and/or `updated: {{DATE}}`
- **Cause:** Note created from Obsidian Templater template, but Templater auto-substitution did not run (e.g., note created via drag-and-drop, copy-paste, or CLI instead of through Obsidian's "New note from template" command)
- **YAML effect:** `{{DATE}}` is parsed as a YAML flow mapping key — an unhashable type — causing a hard parse failure
- **Prevention:** Always create notes via Obsidian's Templater command, or run a post-creation hook that replaces `{{DATE}}` with today's date

## 2. Markdown heading prefix inside frontmatter (`## title:`)

- **Frequency:** 1 file (now fixed: `Templates/Knowledge Extraction Prompt - Fallback.md`)
- **Pattern:** `## title: Some Title` instead of `title: Some Title`
- **Cause:** Manual editing error — likely the author typed a Markdown heading prefix by mistake
- **YAML effect:** The `##` prefix makes the line a YAML block scalar or comment depending on parser; in practice it causes the `title` field to be absent from the parsed dict
- **Prevention:** Validate with `python3 -c "import yaml; yaml.safe_load(open(f).read()[4:f.find('\n---', 4)])"` after manual frontmatter edits

## 3. Missing frontmatter on knowledge notes

- **Frequency:** 8 files (all Jetson-related, all in `Knowledge/Software Engineering/`)
- **Cause:** Notes likely imported or migrated from another source without adding frontmatter
- **Prevention:** Run the quality audit script (in `CLAUDE.md`) after any bulk import

---

# Verification Steps

After applying the manual fixes above, re-run the audit to confirm zero remaining issues:

```bash
cd "/Users/turcinv/Documents/personal_knowledge/Career Knowledge Base"
python3 /tmp/frontmatter_audit.py
```

**Expected output after all fixes applied:**
```
valid:                   1639
no_frontmatter:          2      ← CLAUDE.md and README.md only
unclosed_frontmatter:    0
heading_in_frontmatter:  0
yaml_parse_error:        0
missing_required_fields: 0
domain_mismatch:         0
```

**Verify the original fixed file specifically:**
```bash
python3 -c "
import yaml
with open('Templates/Knowledge Extraction Prompt - Fallback.md', encoding='utf-8') as f:
    content = f.read()
end = content.find('\n---', 4)
fm = yaml.safe_load(content[4:end])
required = {'title','domain','type','created','updated','tags','source','confidence','status'}
missing = required - set(fm.keys())
print('Missing fields:', missing or 'none')
print('title:', fm.get('title'))
"
```

**Clean up `.bak` files once satisfied:**
```bash
find "/Users/turcinv/Documents/personal_knowledge/Career Knowledge Base" -name "*.bak" -delete
echo "Backups removed"
```

**Commit fixes:**
```bash
cd "/Users/turcinv/Documents/personal_knowledge/Career Knowledge Base"
git add -A
git commit -m "Fix frontmatter: resolve Templater {{DATE}} placeholders, add missing frontmatter to Jetson notes"
```

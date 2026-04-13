#!/bin/bash
# rebuild_spec.sh — Rebuilds Hyperion-Spec-v2.1.docx from PDF + Addendum
# Uses Claude Code to read, chunk, and reassemble the full spec with changes baked in.
#
# Usage: cd ~/hyperion && bash rebuild_spec.sh
#
# Prerequisites:
#   - claude CLI installed and authenticated
#   - npm docx package installed (npm install docx)
#   - Hyperion-Spec-v2.pdf and Hyperion-Spec-v2-Addendum.docx in current directory

set -e

SPEC="docs/Hyperion-Spec-v2.pdf"
ADDENDUM="docs/Hyperion-Spec-v2-Addendum.docx"
RESULTS="docs/Hyperion-Phase1-Results.docx"
OUTDIR="spec_build"

# Preflight checks
for f in "$SPEC" "$ADDENDUM"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found in $(pwd)"
    exit 1
  fi
done

if ! command -v claude &> /dev/null; then
  echo "ERROR: claude CLI not found. Install from https://claude.ai/code"
  exit 1
fi

mkdir -p "$OUTDIR"

echo "============================================"
echo "  Hyperion Spec v2.1 Rebuild"
echo "  $(date)"
echo "============================================"
echo ""

# ──────────────────────────────────────────────
# Prompt 0: Scaffolding — shared styles & helpers
# ──────────────────────────────────────────────
echo "[0/9] Scaffolding: document skeleton, styles, helpers..."
claude --dangerously-skip-permissions -p "
You are rebuilding Hyperion-Spec-v2.pdf as a clean .docx file. This is prompt 0 of 9.

Create a file called $OUTDIR/helpers.js that exports:
- Document style config (Arial font, US Letter 12240x15840, 1-inch margins)
- Heading styles: Heading1 (18pt bold, color #2E5090), Heading2 (14pt bold, #2E5090), Heading3 (13pt bold, #2E5090)
- Default paragraph: 12pt Arial, color #1A1A1A, spacing after 200
- Numbering configs for bullets and numbered lists (use LevelFormat.BULLET, never unicode bullets)
- Table helper: function makeTable(headers, rows, columnWidths) that creates a table with #2E5090 header row, white text, CCCCCC borders, cell margins 80/80/120/120, WidthType.DXA, ShadingType.CLEAR
- Paragraph helpers: para(text), boldPara(boldText, normalText), bullet(runs)
- Header: 'Hyperion Technical Specification v2.1' in 9pt Arial #2E5090, bottom border
- Footer: centered page numbers in 9pt Arial #666666

Use: const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber, PageBreak } = require('docx');

Export everything as module.exports = { styles, numbering, makeTable, para, boldPara, bullet, heading, heading2, heading3, pageBreak, header, footer, BLUE, DARK, GRAY, borders, cellMargins }.

Do NOT generate the final document yet. Only create the helpers file.
" --max-turns 15

echo "  ✓ helpers.js created"
echo ""

# ──────────────────────────────────────────────
# Prompt 1: Sections 1-2 (Executive Summary, System Architecture)
# ──────────────────────────────────────────────
echo "[1/9] Sections 1-2: Executive Summary, System Architecture..."
claude --dangerously-skip-permissions -p "
You are rebuilding Hyperion-Spec-v2.pdf as a clean .docx. This is prompt 1 of 9.

Read $SPEC pages 1-5. Read $OUTDIR/helpers.js for the shared styles and helpers.

Create $OUTDIR/sections_01_02.js that exports a function buildSections() returning an array of docx elements (Paragraphs, Tables, PageBreaks).

Include:
- Table of Contents page (just the list of 20 sections, matching the PDF)
- Section 1: Executive Summary — reproduce ALL paragraphs exactly as in the PDF
- Section 2: System Architecture — ALL subsections (2.1 Architecture Overview with the infrastructure table, 2.2 Data Flow with all 7 pipeline stages, 2.3 Language Split with the Python/TypeScript table, 2.4 Multi-Tenant Architecture with all bullet points)

Reproduce every table, every bullet point, every paragraph. Do not summarize or skip content.
Use the helpers from $OUTDIR/helpers.js (require it with relative path).
End with a PageBreak.
" --max-turns 20

echo "  ✓ sections_01_02.js created"
echo ""

# ──────────────────────────────────────────────
# Prompt 2: Sections 3-4 (SAE Pipeline, Vector Storage)
# ──────────────────────────────────────────────
echo "[2/9] Sections 3-4: SAE Pipeline, Vector Storage..."
claude --dangerously-skip-permissions -p "
You are rebuilding Hyperion-Spec-v2.pdf as a clean .docx. This is prompt 2 of 9.

Read $SPEC pages 6-10. Read $ADDENDUM. Read $OUTDIR/helpers.js.

Create $OUTDIR/sections_03_04.js that exports buildSections() returning an array of docx elements.

Include:
- Section 3: Sparse Autoencoder Pipeline — ALL subsections (3.1 Model Selection, 3.2 Feature Filtering Pipeline with the 4-stage table, 3.3 Feature Clustering and Labeling including Delphi steps and Taxonomy, 3.4 Multi-Document Encoding, 3.5 Feature Instability and Versioning)
- Section 4: Vector Storage (Pinecone)
  - 4.1 Hybrid Dense + Sparse Architecture (with the Dense/Sparse comparison table) — FROM THE PDF, unchanged
  - 4.2 Score Fusion Matrix — USE THE REVISED VERSION FROM THE ADDENDUM, NOT THE PDF. Key changes: mandatory gics_sector/gics_industry metadata filter, score fusion operates within industry-filtered candidate set, cross-industry opt-in paragraph added.
  - 4.3 Record Schema (with the metadata fields list) — FROM THE PDF, unchanged

CRITICAL: Section 4.2 MUST come from the addendum. Sections 4.1 and 4.3 come from the PDF.
Use helpers from $OUTDIR/helpers.js. End with PageBreak.
" --max-turns 20

echo "  ✓ sections_03_04.js created"
echo ""

# ──────────────────────────────────────────────
# Prompt 3: Sections 5-6 (Convex Ontology, Query Design)
# ──────────────────────────────────────────────
echo "[3/9] Sections 5-6: Convex Ontology, Query Design..."
claude --dangerously-skip-permissions -p "
You are rebuilding Hyperion-Spec-v2.pdf as a clean .docx. This is prompt 3 of 9.

Read $SPEC pages 11-20. Read $ADDENDUM. Read $OUTDIR/helpers.js.

Create $OUTDIR/sections_05_06.js that exports buildSections() returning an array of docx elements.

Include:
- Section 5: Event-Sourced Ontology (Convex) — FROM THE PDF, unchanged. ALL subsections:
  5.1 Design Philosophy, 5.2 Core Entities (DocumentRecord, CompanyState, StateTransition, AnalogMatch, ExecutiveState, CompetitiveRelationship, PortfolioTransaction, SectorSnapshot, QueryLog — each with their full field tables), 5.3 Entity Relationships, 5.4 Core Query Patterns and Indexes (with the query/access pattern/index table), 5.5 Idempotency
- Section 6: Query Design
  - 6.1 Multi-Document Retrieval Pipeline — USE THE REVISED VERSION FROM THE ADDENDUM. Pipeline is now 5 stages (new Stage 1: Industry Candidate Generation). Original stages renumbered 2-5.
  - 6.2 Five Query Types — USE THE REVISED VERSION FROM THE ADDENDUM. Each query type now specifies its industry filter level.

CRITICAL: Section 5 is unchanged from PDF. Sections 6.1 and 6.2 MUST come from the addendum.
Use helpers from $OUTDIR/helpers.js. End with PageBreak.
" --max-turns 25

echo "  ✓ sections_05_06.js created"
echo ""

# ──────────────────────────────────────────────
# Prompt 4: Sections 7-9 (LLM, Interface, Dashboard)
# ──────────────────────────────────────────────
echo "[4/9] Sections 7-9: LLM Integration, Interface Layer, Web Dashboard..."
claude --dangerously-skip-permissions -p "
You are rebuilding Hyperion-Spec-v2.pdf as a clean .docx. This is prompt 4 of 9.

Read $SPEC pages 21-35. Read $OUTDIR/helpers.js.

Create $OUTDIR/sections_07_09.js that exports buildSections() returning an array of docx elements.

Include ALL content from the PDF, unchanged:
- Section 7: LLM Integration — ALL subsections (7.1 Three Prompt Jobs, 7.2 Tiered Model Strategy, 7.3 Prompt Injection Defense, 7.4 Model Agnosticism, 7.5 Context Management)
- Section 8: Interface Layer — ALL subsections (8.1 Dual Interface Architecture, 8.2 MCP Architecture, and any other subsections)
- Section 9: Web Dashboard — ALL subsections

Reproduce every paragraph, table, bullet point, and code block exactly. Do not summarize.
Use helpers from $OUTDIR/helpers.js. End with PageBreak.
" --max-turns 25

echo "  ✓ sections_07_09.js created"
echo ""

# ──────────────────────────────────────────────
# Prompt 5: Sections 10-13 (Error Handling, Security, Deployment, Data Sources)
# ──────────────────────────────────────────────
echo "[5/9] Sections 10-13: Error Handling, Security, Deployment, Data Sources..."
claude --dangerously-skip-permissions -p "
You are rebuilding Hyperion-Spec-v2.pdf as a clean .docx. This is prompt 5 of 9.

Read $SPEC pages 36-48. Read $OUTDIR/helpers.js.

Create $OUTDIR/sections_10_13.js that exports buildSections() returning an array of docx elements.

Include ALL content from the PDF, unchanged:
- Section 10: Error Handling and Resilience — ALL subsections
- Section 11: Security and Hardening — ALL subsections
- Section 12: Deployment Topology — ALL subsections
- Section 13: Data Sources — ALL subsections

Reproduce every paragraph, table, bullet point exactly. Do not summarize.
Use helpers from $OUTDIR/helpers.js. End with PageBreak.
" --max-turns 25

echo "  ✓ sections_10_13.js created"
echo ""

# ──────────────────────────────────────────────
# Prompt 6: Sections 14-16 (Cost Model, Ralph Loop, Implementation)
# ──────────────────────────────────────────────
echo "[6/9] Sections 14-16: Cost Model, Ralph Loop, Implementation Phases..."
claude --dangerously-skip-permissions -p "
You are rebuilding Hyperion-Spec-v2.pdf as a clean .docx. This is prompt 6 of 9.

Read $SPEC pages 49-58. Read $OUTDIR/helpers.js.

Create $OUTDIR/sections_14_16.js that exports buildSections() returning an array of docx elements.

Include ALL content from the PDF, unchanged:
- Section 14: Cost Model — ALL subsections and tables
- Section 15: Development Methodology — The Ralph Loop — ALL subsections including the five principles, the process description, PLAN.md details, audit agent, back pressure mechanisms
- Section 16: Implementation Phases — ALL subsections (Phase A, B, C, trigger-gated additions in 16.6)

Reproduce every paragraph, table, bullet point exactly. Do not summarize.
Use helpers from $OUTDIR/helpers.js. End with PageBreak.
" --max-turns 25

echo "  ✓ sections_14_16.js created"
echo ""

# ──────────────────────────────────────────────
# Prompt 7: Section 17 (Testing & Validation — REPLACED)
# ──────────────────────────────────────────────
echo "[7/9] Section 17: Testing and Validation (REPLACED with Phase 1 results)..."
claude --dangerously-skip-permissions -p "
You are rebuilding Hyperion-Spec-v2.pdf as a clean .docx. This is prompt 7 of 9.

Read $ADDENDUM and $RESULTS. Read $OUTDIR/helpers.js.

Create $OUTDIR/sections_17.js that exports buildSections() returning an array of docx elements.

Section 17 is COMPLETELY REPLACED. Do NOT use the PDF version. Use the addendum's 'Phase 1 Validation Summary (New Section 17)' section.

Include:
- Section heading: '17. Testing and Validation — Phase 1 Results'
- Context paragraph explaining 13 tests, 3 layers, 25 years, 14.9M pairs
- Results summary table with columns: Test, Result, Verdict, Implication. Color-code verdict cells (green for PASS, red for FAIL, yellow for QUAL PASS).
- Validated Architecture subsection with the three-stage pipeline
- Statistical Methodology subsection with data details, bootstrap methodology, factor model details, controls, significance standards (pull these from the Phase 1 Results doc)
- 1C Permutation Test note (not yet run, not product-gating)

Use helpers from $OUTDIR/helpers.js. End with PageBreak.
" --max-turns 20

echo "  ✓ sections_17.js created"
echo ""

# ──────────────────────────────────────────────
# Prompt 8: Sections 18-20 (Research, Design Decisions, Competitive)
# ──────────────────────────────────────────────
echo "[8/9] Sections 18-20: Research Foundations, Key Design Decisions, Competitive Position..."
claude --dangerously-skip-permissions -p "
You are rebuilding Hyperion-Spec-v2.pdf as a clean .docx. This is prompt 8 of 9.

Read $SPEC pages 59-66. Read $OUTDIR/helpers.js.

Create $OUTDIR/sections_18_20.js that exports buildSections() returning an array of docx elements.

Include ALL content from the PDF, unchanged:
- Section 18: Research Foundations — ALL subsections and citations
- Section 19: Key Design Decisions — ALL subsections and rationale
- Section 20: Competitive Position — ALL subsections and competitor analysis

Reproduce every paragraph, table, bullet point exactly. Do not summarize.
Use helpers from $OUTDIR/helpers.js. No trailing PageBreak (this is the last section).
" --max-turns 25

echo "  ✓ sections_18_20.js created"
echo ""

# ──────────────────────────────────────────────
# Prompt 9: Assemble and validate
# ──────────────────────────────────────────────
echo "[9/9] Assembling final document..."
claude --dangerously-skip-permissions -p "
You are assembling the final Hyperion-Spec-v2.1.docx. This is prompt 9 of 9.

Read all files in $OUTDIR/: helpers.js, sections_01_02.js, sections_03_04.js, sections_05_06.js, sections_07_09.js, sections_10_13.js, sections_14_16.js, sections_17.js, sections_18_20.js.

Create $OUTDIR/assemble.js that:
1. Requires all section files and helpers.js
2. Creates a new Document using the styles, numbering, header, and footer from helpers.js
3. Combines all sections in order into a single sections array
4. Generates Hyperion-Spec-v2.1.docx in the project root directory
5. Prints the file size

Then run it: node $OUTDIR/assemble.js

If any section file has an error, fix it and re-run.
After successful generation, count the approximate pages (total paragraphs / ~40 per page) and report.

The output file should be Hyperion-Spec-v2.1.docx in the current directory.
" --max-turns 25

echo ""
echo "============================================"

if [ -f "Hyperion-Spec-v2.1.docx" ]; then
  SIZE=$(ls -lh Hyperion-Spec-v2.1.docx | awk '{print $5}')
  echo "  ✓ SUCCESS: Hyperion-Spec-v2.1.docx ($SIZE)"
  echo "  Changes baked in:"
  echo "    - Section 4.2: Mandatory industry pre-filter on all queries"
  echo "    - Section 6.1: 5-stage pipeline (new Stage 1: Industry Candidate Gen)"
  echo "    - Section 6.2: Each query type specifies filter level"
  echo "    - Section 17: Replaced with actual Phase 1 results"
else
  echo "  ✗ FAILED: Hyperion-Spec-v2.1.docx not found"
  echo "  Check $OUTDIR/ for partial outputs and errors"
fi

echo "============================================"

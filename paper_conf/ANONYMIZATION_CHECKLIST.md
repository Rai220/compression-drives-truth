# Anonymization Checklist for Conference Submission

## Paper (main.tex)
- [x] No author names (use Anonymous Authors)
- [x] No arXiv link in paper text
- [x] No GitHub repo link in paper text
- [x] Appendix A says "[anonymous repository, included as supplementary material]"
- [ ] Do NOT write "under review at [conference]" on arXiv version

## Supplementary Code ZIP
Before submission, create anonymized code package:
- [ ] Remove .git/ directory
- [ ] Remove README.md (contains author name, LinkedIn, arXiv link)
- [ ] Remove AGENTS.md (contains arXiv link)
- [ ] Remove paper_v2.md (contains author name)
- [ ] Remove "Статья К.К..md"
- [ ] Remove paper_tmlr/ (TMLR submission artifacts)
- [ ] Remove .claude/ directory
- [ ] Keep: training/, training_torch/, data/, analysis/, scripts/, Dockerfile
- [ ] Add anonymous README with reproduction instructions only

## Script to create anonymous ZIP:
```bash
mkdir -p /tmp/anon_submission
rsync -av --exclude='.git' --exclude='README.md' --exclude='AGENTS.md' \
    --exclude='paper_v2.md' --exclude='paper_tmlr' --exclude='.claude' \
    --exclude='Статья К.К..md' --exclude='*.npz' --exclude='*.pt' \
    --exclude='*.safetensors' --exclude='.venv*' --exclude='__pycache__' \
    --exclude='results/' --exclude='TODO.md' --exclude='paper_v3.md' \
    --exclude='modal_run*.py' \
    . /tmp/anon_submission/

# Add anonymous README
cat > /tmp/anon_submission/README.md << 'EOF'
# Supplementary Code

Code for "Error Structure Determines Correctness Preference in Contradictory Training Data"

See scripts/reproduce_minimal.sh for minimal reproduction.
See Dockerfile for containerized reproduction.
EOF

cd /tmp && zip -r anon_submission.zip anon_submission/
```

## Description
<!-- Briefly describe what this PR does and why -->

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup
- [ ] Documentation update
- [ ] Model / training improvement

## Related Issues
Closes #<!-- issue number -->

## Changes Made
<!-- List the key files changed and what was changed -->
- `semantic_neurosymbolic_vqa.py` — 
- `ensemble_vqa_app.py` — 
- `backend_api.py` — 

## Testing Done
- [ ] Ran `python test_vqa_enhancements.py`
- [ ] Manually tested with backend running (`python backend_api.py`)
- [ ] Tested Expo UI (`npx expo start --clear`)
- [ ] Tested neuro-symbolic routing with a reasoning question
- [ ] Tested neural fallback with a visual question

## Checklist
- [ ] No hardcoded object lists or keyword patterns added
- [ ] Wikidata queries use property IDs (P-numbers), not hardcoded Q-IDs
- [ ] Groq is used only for verbalization (not free reasoning)
- [ ] No secrets or API keys committed
- [ ] `requirements_api.txt` updated if new deps added

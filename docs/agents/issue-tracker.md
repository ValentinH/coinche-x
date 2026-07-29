# Issue tracker: GitHub

Issues and planning artifacts live in GitHub Issues. Use the `gh` CLI for every GitHub operation.

## Conventions

- Create: `gh issue create --title "..." --body "..."`
- Read: `gh issue view <number> --comments`
- List: `gh issue list --state open --json number,title,body,labels,assignees`
- Comment: `gh issue comment <number> --body "..."`
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`
- Assign: `gh issue edit <number> --add-assignee "@me"`
- Close: `gh issue close <number> --comment "..."`

Infer the repository from `git remote -v`.

## Pull requests as a triage surface

No. Pull requests are not requests and are excluded from triage.

## Wayfinding operations

- A map is an issue labelled `wayfinder:map`.
- Create tickets with `gh issue create`, then attach each as a native child:
  `coinche_child_id=$(gh api repos/{owner}/{repo}/issues/<child-number> --jq .id)` then
  `gh api --method POST -H "X-GitHub-Api-Version: 2026-03-10" repos/{owner}/{repo}/issues/<map-number>/sub_issues -F sub_issue_id="$coinche_child_id"`.
- Inspect children:
  `gh api --paginate -H "X-GitHub-Api-Version: 2026-03-10" repos/{owner}/{repo}/issues/<map-number>/sub_issues`.
- Create all tickets before wiring dependencies.
- Add a native dependency:
  `coinche_blocker_id=$(gh api repos/{owner}/{repo}/issues/<blocking-number> --jq .id)` then
  `gh api --method POST -H "X-GitHub-Api-Version: 2026-03-10" repos/{owner}/{repo}/issues/<blocked-number>/dependencies/blocked_by -F issue_id="$coinche_blocker_id"`.
- Inspect blockers:
  `gh api --paginate -H "X-GitHub-Api-Version: 2026-03-10" repos/{owner}/{repo}/issues/<number>/dependencies/blocked_by`.
- The frontier is the map's open, unassigned child issues whose `blockedBy` issues are all closed, preserving child order.
- Claim before work: `gh issue edit <number> --add-assignee "@me"`

When a skill says to publish, create a GitHub issue.

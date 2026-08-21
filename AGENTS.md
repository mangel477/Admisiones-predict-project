# Changelog Maintenance

Maintain `CHANGELOG.md` as the user-facing record of meaningful project changes.

- Inspect the actual diff and relevant Git history before writing an entry.
- Add user-notable changes to `## [Unreleased]` under Keep a Changelog categories such as `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, or `Security`.
- Consolidate related work into outcome-focused entries; do not duplicate individual commits.
- Exclude internal implementation detail, dependency maintenance, formatting-only changes, and other noise unless users are materially affected.
- Move entries from `Unreleased` only for an explicit release. Use a valid Semantic Version and its real release date.
- Keep newest release sections first and preserve stable Markdown headings and link references.
- Never invent facts, versions, dates, repository links, comparison links, capabilities, or verification outcomes.
- always write in spanish for the changelog.

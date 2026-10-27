# Preparing FP Play v0.2.0-rc.1

Milestone 14 prepares **v0.2.0-rc.1**, a SemVer prerelease for the subsequent
numerical/hardware work. The existing `v0.1.0` tag points to `977ccc8` (Milestone 7)
and remains unchanged. This document supersedes the old “0.1.0 is untagged”
guidance; it does not assert whether a GitHub release or Pages deployment exists.
`private: true` intentionally prevents accidental npm registry publication.
Package/lockfile and the visible UI derive the same candidate version.

No new tag, GitHub release, deployment, repository-setting change, push or merge
is authorized by release **preparation**. Candidate acceptance requires review of
CI and the remaining browser/accessibility limits in the verification record.

## Local production verification

```sh
npm ci
npm run check
npm run check:release
npm run preview
```

Open `http://localhost:4173/fp-play/`. Verify the [manual release cases](RELEASE_VERIFICATION.md), including reloads of `#explorer`, `#microscope`, `#arithmetic`, `#fma`, `#research`, `#hardware`, and `#datapath`. The FMA hash loads a verified comparison with original FP16 operands. Hash navigation keeps all URLs on the static index; `/fp-play/arithmetic` is not an application route. There is no router, server rewrite, or manufactured 404 fallback.

Vite's base is `/fp-play/` for both development and production. The expected project-site address is `https://nivupai.github.io/fp-play/`; this is a deployment target, not a claim that it is live. A renamed repository, organization site, or custom-domain root requires reviewing the base path and release asset checks together.

## Explicit maintainer steps to deploy

Only do these when publication is authorized:

1. Review the feature branch and its CI results, then merge the approved change into `main`. The workflow file must be on the default branch to appear in Actions.
2. In the GitHub repository, open **Settings → Pages**. Under **Build and deployment**, set **Source** to **GitHub Actions**. This is a manual repository-settings change; the workflow has `enablement: false` and cannot perform it.
3. Open **Actions → Deploy GitHub Pages (manual) → Run workflow**, select **main**, and explicitly run it. Other refs skip the build/deploy chain. There are no push, pull-request, tag, release, or `workflow_run` deployment triggers.
4. The build job installs the lockfile, runs the complete checks/build and production path verification, reads Pages metadata, and uploads only `dist/`. The dependent deploy job uses the `github-pages` environment and OIDC. Any environment approval rules configured by maintainers remain in effect.
5. After successful deployment, use the URL reported by the deploy job. Repeat the navigation, asset, responsive, theme, and numerical-example checks against that actual URL. Confirm the documented source/technical links work after merge.

The [Vite Pages guide](https://vite.dev/guide/static-deploy.html#github-pages) explains the repository base, and [GitHub's custom workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) describes Pages permissions and artifacts.

## Workflow permissions and behavior

The ordinary CI workflow is read-only and runs all tests on Node 22.9.0 and 24. It cannot deploy. The separate Pages workflow is manual and main-only. Its build job has `contents: read` and `pages: read`; only its dependent deploy job has `pages: write` and `id-token: write`. Checkout does not persist credentials. No PAT or application secret is required.

Official actions are pinned to verified release commit IDs, with versions in comments. Pages configuration is read rather than enabled. Artifacts retain one day and deployments are serialized without canceling an in-progress publication. Validation failure prevents artifact upload/deployment. Review permissions and triggers whenever updating the workflow; never add an automatic publishing trigger without explicit authorization.

## Publishing the source release

After explicit authorization and successful review, a maintainer can create the `v0.2.0-rc.1` candidate tag and prerelease from the reviewed `main` commit. Mark the candidate as a prerelease, not a stable release. Use [CHANGELOG.md](../CHANGELOG.md) for release notes and accurately record browser verification and known limitations. Do not describe the finite verification corpus as a proof of full IEEE 754 conformance.

If rollback is necessary, revert the relevant source change through the normal review process on `main`, then explicitly rerun the manual Pages workflow. No rollback or release publication is performed by local scripts.

## Candidate acceptance checklist

- Confirm version `0.2.0-rc.1` in package/lockfile, built UI, changelog and notes.
- Preserve the 502 historical tests and inspect new showcase/artifact tests.
- Run full fast checks, supported Node versions, release paths/notices/downloads,
  and relevant pinned RTL checks; record actual counts/tools.
- Review case-study numbers against the original experiment artifacts and retain
  source hashes, dirty measurement state, conditions and scope limits.
- Recheck desktop/mobile themes, keyboard/chart/wave controls and the remaining
  screen-reader/device checklist. No full WCAG/IEEE conformance claim.
- Preserve third-party notices in the static output and external library/tool
  license boundaries. New GitHub document links become available after merge.
- Obtain explicit authorization before merge, settings, workflow dispatch, tag or
  GitHub prerelease creation. A future stable `0.2.0` requires its own reviewed
  version change and authorization; no release script automatically promotes it.

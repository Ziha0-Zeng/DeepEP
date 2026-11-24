// Copyright (c) 2026, Lu Lu
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { request, validateReport, expectedStages, markPending, prepare, finalize } = require('../github.cjs');

const env = {
  CI_PR: '8', CI_MODE: 'quick', CI_HEAD_SHA: 'a'.repeat(40), CI_BASE_SHA: 'b'.repeat(40),
  CI_TRUSTED_SHA: 'b'.repeat(40), CI_BASE_REF: 'main', GITHUB_RUN_ID: '100', GITHUB_RUN_ATTEMPT: '2',
  CI_A5_RESULT: 'success', CI_DOWNLOAD_RESULT: 'success',
};
function report() {
  return {
    schema: 1, status: 'success', pr: '8', mode: 'quick', world: 2,
    head_sha: env.CI_HEAD_SHA, base_sha: env.CI_BASE_SHA, trusted_sha: env.CI_TRUSTED_SHA,
    run_id: '100', run_attempt: '2', wheel_sha256: crypto.createHash('sha256').update('test wheel').digest('hex'), source_sha256: 'e'.repeat(64),
    stages: expectedStages(2).map(name => ({ name, status: 'success' })),
  };
}

test('reject malformed dispatch parameters', () => {
  for (const CI_PR of ['', '-1', '1;echo', '1.0', '9007199254740992']) {
    assert.throws(() => request({ ...env, CI_PR }));
  }
  assert.throws(() => request({ ...env, CI_MODE: 'scoped' }));
});

test('reject stale, partial and diagnostic-only reports', () => {
  validateReport(report(), env);
  for (const mutation of [
    { head_sha: 'old' }, { run_attempt: '1' }, { mode: 'full' }, { status: 'failure' },
    { stages: [] }, { wheel_sha256: '' }, { world: 8 }, { schema: 2 },
  ]) assert.throws(() => validateReport({ ...report(), ...mutation }, env));
  const failed = report();
  failed.stages[1].status = 'skipped';
  assert.throws(() => validateReport(failed, env));
});

function mocks() {
  const statuses = [];
  const failures = [];
  const outputs = {};
  const pr = {
    state: 'open', draft: false, html_url: 'https://github.com/owner/DeepEP/pull/8',
    head: { sha: env.CI_HEAD_SHA, repo: { full_name: 'fork/DeepEP' } }, base: { ref: 'main' },
  };
  const github = { rest: {
    pulls: { get: async () => ({ data: pr }) },
    repos: {
      getCollaboratorPermissionLevel: async () => ({ data: { permission: 'admin' } }),
      get: async () => ({ data: { default_branch: 'main' } }),
      getBranch: async () => ({ data: { commit: { sha: env.CI_BASE_SHA } } }),
      getCombinedStatusForRef: async () => ({ data: { statuses: [] } }),
      createCommitStatus: async s => statuses.push(s),
    },
  } };
  const core = {
    setFailed: message => failures.push(message), setOutput: (k, v) => { outputs[k] = v; },
    summary: { addHeading() { return this; }, addRaw() { return this; }, addTable() { return this; }, async write() {} },
  };
  const context = {
    actor: 'admin', ref: 'refs/heads/main', repo: { owner: 'owner', repo: 'DeepEP' },
    serverUrl: 'https://github.com', runId: 100, payload: { pull_request: pr },
  };
  return { github, core, context, pr, statuses, failures, outputs, env };
}

test('initialize only missing contexts on the PR head', async () => {
  const m = mocks();
  m.github.rest.repos.getCombinedStatusForRef = async () => ({
    data: { statuses: [{ context: 'DeepEP / A5 HT', state: 'success' }] },
  });
  await markPending(m);
  assert.equal(m.statuses.length, 3);
  assert.ok(m.statuses.every(status => status.sha === env.CI_HEAD_SHA && status.state === 'pending'));
  assert.ok(m.statuses.every(status => status.context !== 'DeepEP / A5 HT'));
});

test('only admin and default branch definitions can queue runners', async () => {
  const m = mocks();
  await prepare(m);
  assert.equal(m.outputs.head_sha, env.CI_HEAD_SHA);
  assert.equal(m.outputs.base_ref, 'main');
  assert.equal(m.statuses.length, 4);
  assert.equal(m.statuses[0].state, 'pending');
  m.github.rest.repos.getCollaboratorPermissionLevel = async () => ({ data: { permission: 'write' } });
  await assert.rejects(prepare(m), /admin/);
  const branch = mocks();
  branch.context.ref = 'refs/heads/pr-changes';
  await assert.rejects(prepare(branch), /default branch/);
  const stale = mocks();
  stale.env = { ...env, CI_TRUSTED_SHA: 'c'.repeat(40) };
  await assert.rejects(prepare(stale), /Default branch changed/);
  const rerun = mocks();
  rerun.env = { ...env, CI_TRIGGERING_ACTOR: 'contributor' };
  rerun.github.rest.repos.getCollaboratorPermissionLevel = async ({ username }) => ({
    data: { permission: username === 'admin' ? 'admin' : 'write' },
  });
  await assert.rejects(prepare(rerun), /admin/);
  const comment = mocks();
  comment.context.actor = 'github-actions[bot]';
  comment.env = { ...env, CI_TRIGGERING_ACTOR: 'github-actions[bot]', CI_REQUESTED_BY: 'admin' };
  comment.github.rest.repos.getCollaboratorPermissionLevel = rerun.github.rest.repos.getCollaboratorPermissionLevel;
  await prepare(comment);
  assert.equal(comment.outputs.head_sha, env.CI_HEAD_SHA);
});

function writeArtifacts(workspace, result) {
  const directory = path.join(workspace, 'reports');
  fs.writeFileSync(path.join(directory, 'result.json'), JSON.stringify(result));
  fs.mkdirSync(path.join(directory, 'package/dist'), { recursive: true });
  fs.writeFileSync(path.join(directory, 'package/dist/test.whl'), 'test wheel');
  fs.writeFileSync(path.join(directory, 'package/validation.json'), JSON.stringify({
    wheel_sha256: result.wheel_sha256, runtime_ranks: result.world,
    build_info: { source_sha256: result.source_sha256 },
  }));
  for (const stage of expectedStages(result.world).slice(2)) {
    const stageDir = path.join(directory, 'ht', stage);
    fs.mkdirSync(stageDir, { recursive: true });
    for (let rank = 0; rank < result.world; rank++) {
      fs.writeFileSync(path.join(stageDir, `rank-${rank}.json`), JSON.stringify({
        rank, world_size: result.world, passed: true, checks: 1, source_sha256: result.source_sha256,
        dispatch_checked: true, notify_checked: true,
      }));
    }
  }
}

test('publish failures for missing artifacts, canceled jobs and changed PRs', async () => {
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'deepep-gate-'));
  try {
    fs.mkdirSync(path.join(workspace, 'reports'));
    const m = mocks();
    m.env = { ...env, GITHUB_WORKSPACE: workspace };
    await finalize(m);
    assert.equal(m.statuses.at(-1).state, 'failure');
    writeArtifacts(workspace, report());
    await finalize(m);
    assert.ok(m.statuses.slice(-4).every(status => status.state === 'success'));
    assert.equal(m.statuses.at(-1).context, 'DeepEP / A5 HT');
    m.env.CI_A5_RESULT = 'cancelled';
    await finalize(m);
    assert.equal(m.statuses.at(-1).state, 'failure');
    m.env.CI_A5_RESULT = 'success';
    m.pr.head.sha = 'f'.repeat(40);
    await finalize(m);
    assert.equal(m.statuses.at(-1).state, 'failure');
    assert.equal(m.statuses.at(-1).sha, env.CI_HEAD_SHA);
    m.pr.head.sha = env.CI_HEAD_SHA;
    m.pr.base.ref = 'release';
    await finalize(m);
    assert.equal(m.statuses.at(-1).state, 'failure');
    m.pr.base.ref = 'main';
    const rankReport = path.join(workspace, 'reports/ht/combine-server-fp8/rank-1.json');
    const rank = JSON.parse(fs.readFileSync(rankReport, 'utf8'));
    fs.writeFileSync(rankReport, JSON.stringify({ ...rank, notify_checked: false }));
    await finalize(m);
    assert.match(m.statuses.at(-1).description, /Invalid rank evidence/);
    fs.rmSync(rankReport);
    await finalize(m);
    assert.match(m.statuses.at(-1).description, /rank report count/);
    writeArtifacts(workspace, report());
    fs.writeFileSync(path.join(workspace, 'reports/package/dist/test.whl'), 'wrong wheel');
    await finalize(m);
    assert.match(m.statuses.at(-1).description, /checksum mismatch/);
  } finally {
    fs.rmSync(workspace, { recursive: true, force: true });
  }
});

test('report the failed stage while preserving completed diagnostics', async () => {
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), 'deepep-gate-'));
  try {
    fs.mkdirSync(path.join(workspace, 'reports'));
    const result = report();
    result.status = 'failure';
    result.stages = result.stages.slice(0, 3);
    result.stages[2].status = 'failure';
    writeArtifacts(workspace, result);
    const m = mocks();
    m.env = { ...env, CI_A5_RESULT: 'failure', GITHUB_WORKSPACE: workspace };
    await finalize(m);
    assert.deepEqual(m.statuses.map(status => status.state), ['success', 'failure', 'failure', 'failure']);
    assert.match(m.statuses[1].description, /Failed at dispatch-server/);
    assert.match(m.statuses[2].description, /0\/3 stages completed/);
  } finally {
    fs.rmSync(workspace, { recursive: true, force: true });
  }
});

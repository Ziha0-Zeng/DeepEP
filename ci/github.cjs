// Copyright (c) 2026, Lu Lu
const fs = require('node:fs');
const crypto = require('node:crypto');
const gate = 'DeepEP / A5 HT';
const stageGroups = [
  ['DeepEP / A5 / Environment and Package', /^(environment|package)$/],
  ['DeepEP / A5 / Dispatch', /^dispatch-/],
  ['DeepEP / A5 / Combine', /^combine-/],
];
const contexts = [gate, ...stageGroups.map(([name]) => name)];

function request(env) {
  if (!/^[1-9][0-9]*$/.test(env.CI_PR || '') || !Number.isSafeInteger(Number(env.CI_PR))) {
    throw new Error('Invalid PR number');
  }
  if (!['quick', 'full'].includes(env.CI_MODE)) throw new Error('Invalid CI mode');
  return { pull_number: Number(env.CI_PR), mode: env.CI_MODE };
}

function expectedStages(world) {
  if (![2, 4, 8].includes(world)) throw new Error('Expected EP2, EP4 or EP8');
  return ['environment', 'package', ...['server', ...(world === 8 ? ['rank'] : [])].flatMap(dedup => [
    `dispatch-${dedup}`, `combine-${dedup}-bf16`, `combine-${dedup}-fp8`, `combine-${dedup}-no-weights`,
  ])];
}

function validateProgress(report, env) {
  const expected = {
    schema: 1, head_sha: env.CI_HEAD_SHA, base_sha: env.CI_BASE_SHA,
    trusted_sha: env.CI_TRUSTED_SHA, run_id: env.GITHUB_RUN_ID,
    run_attempt: env.GITHUB_RUN_ATTEMPT, mode: env.CI_MODE, pr: env.CI_PR,
  };
  for (const [key, value] of Object.entries(expected)) {
    if (report[key] !== value) throw new Error(`Report identity mismatch: ${key}`);
  }
  if (!['success', 'failure'].includes(report.status)) throw new Error('Invalid NPU report result');
  const stages = expectedStages(report.world);
  if (!Array.isArray(report.stages) || report.stages.length > stages.length ||
      report.stages.some((s, i) => s?.name !== stages[i] ||
        (s.status !== 'success' && !(s.status === 'failure' && i === report.stages.length - 1 &&
          report.status === 'failure')))) {
    throw new Error('Invalid or out-of-order NPU stages');
  }
}

function validateReport(report, env) {
  validateProgress(report, env);
  if (report.status !== 'success') throw new Error('NPU execution did not succeed');
  if (!/^[a-f0-9]{64}$/.test(report.wheel_sha256 || '') ||
      !/^[a-f0-9]{64}$/.test(report.source_sha256 || '')) throw new Error('Missing artifact identity');
  if (report.stages.length !== expectedStages(report.world).length) {
    throw new Error('Required NPU stages are missing');
  }
}

function readReport(filename) {
  const stat = fs.lstatSync(filename);
  if (!stat.isFile() || stat.size > 1024 * 1024) throw new Error('Invalid report file type or size');
  return JSON.parse(fs.readFileSync(filename, 'utf8'));
}

function validateRanks(root, report) {
  for (const stage of expectedStages(report.world).slice(2)) {
    const directory = `${root}/ht/${stage}`;
    const reports = fs.readdirSync(directory).filter(name => /^rank-.*\.json$/.test(name));
    if (reports.length !== report.world) throw new Error(`Wrong rank report count for ${stage}`);
    for (let rank = 0; rank < report.world; rank++) {
      const data = readReport(`${directory}/rank-${rank}.json`);
      if (data.rank !== rank || data.world_size !== report.world || data.passed !== true ||
          !Number.isInteger(data.checks) || data.checks < 1 || data.source_sha256 !== report.source_sha256 ||
          (stage.startsWith('combine-') && (data.dispatch_checked !== true || data.notify_checked !== true))) {
        throw new Error(`Invalid rank evidence for ${stage}, rank ${rank}`);
      }
    }
  }
}

async function validateArtifacts(root, report) {
  const directory = `${root}/package/dist`;
  const wheels = fs.readdirSync(directory).filter(name => name.endsWith('.whl'));
  if (wheels.length !== 1) throw new Error('Expected exactly one wheel artifact');
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(`${directory}/${wheels[0]}`)) hash.update(chunk);
  if (hash.digest('hex') !== report.wheel_sha256) throw new Error('Archived wheel checksum mismatch');
  const validation = readReport(`${root}/package/validation.json`);
  if (validation.wheel_sha256 !== report.wheel_sha256 ||
      validation.build_info?.source_sha256 !== report.source_sha256 || validation.runtime_ranks !== report.world) {
    throw new Error('Package validation identity mismatch');
  }
  validateRanks(root, report);
}

async function markPending({ github, context }) {
  const pr = context.payload.pull_request;
  const { data } = await github.rest.repos.getCombinedStatusForRef({
    ...context.repo, ref: pr.head.sha, per_page: 100,
  });
  const initialized = new Set(data.statuses.map(status => status.context));
  for (const name of contexts.filter(name => !initialized.has(name))) {
    await github.rest.repos.createCommitStatus({
      ...context.repo, sha: pr.head.sha, context: name, state: 'pending', target_url: pr.html_url,
      description: 'Awaiting admin /run-npu-ci quick or full',
    });
  }
}

function runUrl(context) {
  return `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`;
}

async function prepare({ github, context, core, env }) {
  const { pull_number } = request(env);
  const actor = env.CI_TRIGGERING_ACTOR || context.actor;
  const username = actor === 'github-actions[bot]' ? env.CI_REQUESTED_BY : actor;
  if (!username) throw new Error('A repository admin must request the A5 run');
  const { data: permission } = await github.rest.repos.getCollaboratorPermissionLevel({ ...context.repo, username });
  if (permission.permission !== 'admin') throw new Error('Repository admin permission is required');
  const { data: repo } = await github.rest.repos.get(context.repo);
  if (context.ref !== `refs/heads/${repo.default_branch}`) throw new Error('Run the default branch CI definition');
  const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number });
  if (pr.state !== 'open' || pr.draft || !pr.head.repo || pr.base.ref !== repo.default_branch) {
    throw new Error('Expected an open, ready PR targeting the default branch');
  }
  const { data: branch } = await github.rest.repos.getBranch({ ...context.repo, branch: pr.base.ref });
  for (const sha of [pr.head.sha, branch.commit.sha]) {
    if (!/^[a-f0-9]{40}$/.test(sha)) throw new Error('Invalid source revision');
  }
  if (branch.commit.sha !== env.CI_TRUSTED_SHA) {
    throw new Error('Default branch changed before validation; start a new run');
  }
  for (const name of contexts) {
    await github.rest.repos.createCommitStatus({
      ...context.repo, sha: pr.head.sha, context: name, state: 'pending',
      description: 'A5 HT build and correctness checks queued', target_url: runUrl(context),
    });
  }
  core.setOutput('head_sha', pr.head.sha);
  core.setOutput('head_repo', pr.head.repo.full_name);
  core.setOutput('base_sha', branch.commit.sha);
  core.setOutput('base_ref', pr.base.ref);
}

function stageResults(report, passed) {
  const stages = report ? expectedStages(report.world) : [];
  const completed = new Map(report?.stages.map(stage => [stage.name, stage.status]));
  return stageGroups.map(([name, pattern]) => {
    const required = stages.filter(stage => pattern.test(stage));
    const failed = required.find(stage => completed.get(stage) === 'failure');
    const count = required.filter(stage => completed.get(stage) === 'success').length;
    const success = passed || (report?.status === 'failure' && count === required.length);
    let description = 'No verified A5 result';
    if (success) description = `${count}/${required.length} stages passed`;
    else if (failed) description = `Failed at ${failed}`;
    else if (report?.status === 'failure') description = `${count}/${required.length} stages completed`;
    return { context: name, state: success ? 'success' : 'failure', description };
  });
}

async function finalize({ github, context, core, env }) {
  let state = 'failure';
  let description;
  let report;
  try {
    const { pull_number } = request(env);
    if (env.CI_DOWNLOAD_RESULT !== 'success') throw new Error('A5 artifact download did not succeed');
    const directory = `${env.GITHUB_WORKSPACE}/reports`;
    const candidate = readReport(`${directory}/result.json`);
    validateProgress(candidate, env);
    report = candidate;
    if (env.CI_A5_RESULT !== 'success') {
      const failed = report.stages.find(stage => stage.status === 'failure');
      throw new Error(failed ? `A5 failed at ${failed.name}` : 'A5 job did not succeed');
    }
    validateReport(report, env);
    await validateArtifacts(directory, report);
    const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number });
    if (pr.state !== 'open' || pr.draft || pr.head.sha !== env.CI_HEAD_SHA || pr.base.ref !== env.CI_BASE_REF) {
      throw new Error('PR changed during CI; run again after updating the PR');
    }
    const { data: base } = await github.rest.repos.getBranch({ ...context.repo, branch: env.CI_BASE_REF });
    if (base.commit.sha !== env.CI_BASE_SHA) throw new Error('Base changed during CI; update the PR and run again');
    state = 'success';
    description = `A5 EP${report.world} HT ${env.CI_MODE}: package and all required cases passed`;
  } catch (error) {
    description = error.message;
    core.setFailed(description);
  }
  const stages = stageResults(report, state === 'success');
  await core.summary.addHeading('DeepEP A5 HT').addRaw(`${state}: ${description}`, true).addTable([
    ['Check', 'Result', 'Details'],
    ...stages.map(result => [result.context, result.state, result.description]),
  ]).write();
  // Publish the required aggregate only after all diagnostics are recorded.
  for (const status of [...stages, { context: gate, state, description: description.slice(0, 140) }]) {
    await github.rest.repos.createCommitStatus({
      ...context.repo, sha: env.CI_HEAD_SHA, ...status, target_url: runUrl(context),
    });
  }
}

module.exports = { markPending, prepare, finalize, request, validateReport, validateArtifacts, expectedStages };

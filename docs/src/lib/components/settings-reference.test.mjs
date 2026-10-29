import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { buildSettingsSearchIndex, formatSettingType, summarizeDescription } from './settings-reference.mjs';

describe('formatSettingType', () => {
  it('drops Python class wrappers, typing prefixes and module paths', () => {
    assert.equal(formatSettingType("<class 'pathlib.Path'>"), 'Path');
    assert.equal(formatSettingType("<class 'bool'>"), 'bool');
    assert.equal(formatSettingType('typing.Optional[pathlib.Path]'), 'Optional[Path]');
    assert.equal(
      formatSettingType('typing.Optional[list[invokeai.app.services.config.config_default.URLRegexTokenPair]]'),
      'Optional[list[URLRegexTokenPair]]',
    );
  });
});

describe('summarizeDescription', () => {
  it('keeps the first sentence without splitting abbreviations or code', () => {
    assert.equal(summarizeDescription('IP address to bind to. Use `0.0.0.0` to serve to your network.'), 'IP address to bind to.');
    assert.equal(
      summarizeDescription('Public base path, e.g. `/invoke`. Required when the proxy keeps it.'),
      'Public base path, e.g. `/invoke`.',
    );
    assert.equal(summarizeDescription('No full stop'), 'No full stop');
  });

  it('cuts a long sentence at a word boundary', () => {
    const summary = summarizeDescription(`${'word '.repeat(60)}end.`);
    assert.ok(summary.length <= 160, summary);
    assert.match(summary, /word…$/);
  });
});

describe('buildSettingsSearchIndex', () => {
  it('keeps what a search result shows and the page the results link to', () => {
    const index = buildSettingsSearchIndex(
      [
        {
          name: 'allow_methods',
          type: 'list[str]',
          default: ['*'],
          description: 'Methods allowed for CORS. More detail.',
          env_var: 'INVOKEAI_ALLOW_METHODS',
          category: 'WEB',
          literal_values: [],
          validation: {},
        },
        { name: 'ssl_keyfile', type: 'typing.Optional[pathlib.Path]', default: null, description: 'SSL key file.' },
      ],
      '/configuration/invokeai-yaml/',
    );
    assert.deepEqual(index, {
      page: '/configuration/invokeai-yaml/',
      settings: [
        {
          key: 'allow_methods',
          type: 'list[str]',
          default: '["*"]',
          description: 'Methods allowed for CORS.',
          env: 'INVOKEAI_ALLOW_METHODS',
        },
        { key: 'ssl_keyfile', type: 'Optional[Path]', default: 'null', description: 'SSL key file.' },
      ],
    });
  });
});

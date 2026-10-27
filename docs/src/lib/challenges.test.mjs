import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { createRequire } from 'node:module';
import { join, posix } from 'node:path';
import { after, describe, it } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  adjacentChallenges,
  formatChallengeDates,
  orderWinners,
  repeatedPlaces,
  sortChallenges,
  winnerAlt,
} from './challenges.mjs';

/** @param {string} id @param {number} week @param {string} endDate */
const challenge = (id, week, endDate) => ({ id, data: { week, endDate: new Date(endDate) } });

describe('sortChallenges', () => {
  it('puts the latest end date first, whatever the week numbers say', () => {
    const challenges = [challenge('a', 12, '2026-10-06'), challenge('b', 3, '2026-10-20'), challenge('c', 13, '2026-10-13')];
    assert.deepEqual(
      sortChallenges(challenges).map(({ id }) => id),
      ['b', 'c', 'a'],
    );
  });

  it('orders challenges ending the same day by week, highest first, then by id', () => {
    const challenges = [challenge('x', 4, '2026-10-20'), challenge('z', 5, '2026-10-20'), challenge('y', 5, '2026-10-20')];
    assert.deepEqual(
      sortChallenges(challenges).map(({ id }) => id),
      ['y', 'z', 'x'],
    );
  });

  it('leaves the given array as it was', () => {
    const challenges = [challenge('a', 1, '2026-01-01'), challenge('b', 2, '2026-02-01')];
    sortChallenges(challenges);
    assert.deepEqual(
      challenges.map(({ id }) => id),
      ['a', 'b'],
    );
  });
});

describe('orderWinners', () => {
  it('puts 1st, 2nd and 3rd first, and keeps honourable mentions in file order after them', () => {
    const winners = [
      { place: 'honourable', artist: 'h1' },
      { place: 3, artist: 'third' },
      { place: 'honourable', artist: 'h2' },
      { place: 1, artist: 'first' },
      { place: 2, artist: 'second' },
    ];
    assert.deepEqual(
      orderWinners(winners).map(({ artist }) => artist),
      ['first', 'second', 'third', 'h1', 'h2'],
    );
  });
});

describe('repeatedPlaces', () => {
  it('names each place from 1 to 3 that more than one winner holds', () => {
    const places = [1, 2, 2, 3, 3, 3].map((place) => ({ place }));
    assert.deepEqual(repeatedPlaces(places), [2, 3]);
  });

  it('allows any number of honourable mentions', () => {
    const places = [1, 'honourable', 'honourable', 'honourable'].map((place) => ({ place }));
    assert.deepEqual(repeatedPlaces(places), []);
  });
});

describe('adjacentChallenges', () => {
  const sorted = [challenge('w3', 3, '2026-03-01'), challenge('w2', 2, '2026-02-01'), challenge('w1', 1, '2026-01-01')];

  it('finds the week before and the week after', () => {
    const { older, newer } = adjacentChallenges(sorted, 'w2');
    assert.equal(older?.id, 'w1');
    assert.equal(newer?.id, 'w3');
  });

  it('has no newer challenge for the latest, and no older one for the first', () => {
    assert.equal(adjacentChallenges(sorted, 'w3').newer, undefined);
    assert.equal(adjacentChallenges(sorted, 'w1').older, undefined);
  });

  it('fails for an unknown id rather than linking the wrong weeks', () => {
    assert.throws(() => adjacentChallenges(sorted, 'w9'), /w9/);
  });
});

describe('winnerAlt', () => {
  it('names the title and the artist', () => {
    assert.equal(winnerAlt({ title: 'Tidewater Lantern', artist: 'Ana' }, 44), 'Tidewater Lantern by Ana');
  });

  it('names the week for an untitled entry', () => {
    assert.equal(winnerAlt({ artist: 'mira_k' }, 44), 'Week 44 winner by mira_k');
  });
});

describe('formatChallengeDates', () => {
  // ICU versions differ in the spaces around the dash.
  /** @param {string} start @param {string} end */
  const format = (start, end) => formatChallengeDates(new Date(start), new Date(end)).replace(/\s+/g, ' ');

  it('shows the calendar days YAML dates name', () => {
    // YAML dates are midnight UTC: a local-time format would show October 20 west of Greenwich.
    assert.equal(format('2026-10-21', '2026-10-27'), 'October 21 – 27, 2026');
  });

  it('shows both months, and both years when they differ', () => {
    assert.equal(format('2026-10-28', '2026-11-03'), 'October 28 – November 3, 2026');
    assert.equal(format('2026-12-30', '2027-01-05'), 'December 30, 2026 – January 5, 2027');
  });
});

// The challenge folders are written by hand, so these checks cover what the schema can't: files beside
// `challenge.yaml`, image formats and sizes, folder names, and week numbers across folders.

// The YAML parser Astro's content loaders use, so `challenge.yaml` reads here as it does in the build.
const { parseYaml } = await import(
  pathToFileURL(createRequire(import.meta.resolve('astro')).resolve('@astrojs/internal-helpers/yaml')).href
);

const MAX_IMAGE_BYTES = 500_000;
const IMAGE_EXTENSION = /\.(webp|jpe?g)$/i;
/** `<year>-w<week>-<theme>`, such as `2026-w44-bioluminescence`: the folder name is the page's URL. */
const FOLDER_NAME = /^(\d{4})-w(\d+)-[a-z0-9]+(?:-[a-z0-9]+)*$/;

/**
 * The file a winner's `image` names in its folder, as Astro resolves it (relative to `challenge.yaml`, with or without
 * `./`), or a problem with it.
 * @param {unknown} image
 * @returns {{ file: string } | { problem: string }}
 */
const imageFile = (image) => {
  if (typeof image !== 'string' || image === '') return { problem: 'a winner has no `image`.' };
  if (/^([a-z][a-z\d+.-]*:|\/)/i.test(image)) {
    return { problem: `\`image: ${image}\` must name a file in this folder, such as ./1-artist.webp.` };
  }
  const file = posix.normalize(image);
  if (file.includes('/')) {
    return { problem: `\`image: ${image}\` is outside this folder. Put the image beside challenge.yaml.` };
  }
  return { file };
};

/**
 * What is wrong with a challenges directory, one message per problem. Hidden files, such as `.DS_Store`, are ignored.
 * @param {string} directory
 */
const challengeFolderProblems = (directory) => {
  const problems = [];
  /** @type {Map<string, string[]>} folders by `<start year> week <week>` */
  const foldersByWeek = new Map();
  const visible = (name) => !name.startsWith('.');
  for (const entry of readdirSync(directory, { withFileTypes: true }).filter(({ name }) => visible(name))) {
    if (entry.isFile() && entry.name === 'README.md') continue;
    if (!entry.isDirectory()) {
      problems.push(`${entry.name}: only challenge folders and the README belong here.`);
      continue;
    }
    const folder = entry.name;
    const [, folderYear, folderWeek] = FOLDER_NAME.exec(folder) ?? [];
    if (!folderWeek) {
      problems.push(`${folder}: name the folder <year>-w<week>-<theme>, in lowercase letters, digits and hyphens.`);
    }
    const files = readdirSync(join(directory, folder)).filter(visible);
    if (!files.includes('challenge.yaml')) {
      problems.push(`${folder}: has no challenge.yaml, so the site skips it.`);
      continue;
    }
    let challenge;
    try {
      challenge = parseYaml(readFileSync(join(directory, folder, 'challenge.yaml'), 'utf8')) ?? {};
    } catch (error) {
      problems.push(`${folder}/challenge.yaml: ${error.message}`);
      continue;
    }
    const { week, startDate, winners } = challenge;
    const startYear = startDate instanceof Date ? startDate.getUTCFullYear() : undefined;
    if (folderWeek && Number.isInteger(week) && Number(folderWeek) !== week) {
      problems.push(`${folder}: the folder says week ${Number(folderWeek)}, but challenge.yaml says \`week: ${week}\`.`);
    }
    if (folderYear && startYear && Number(folderYear) !== startYear) {
      problems.push(`${folder}: the folder says ${folderYear}, but the challenge started in ${startYear}.`);
    }
    if (Number.isInteger(week) && startYear) {
      const key = `${startYear} week ${week}`;
      foldersByWeek.set(key, [...(foldersByWeek.get(key) ?? []), folder]);
    }

    const named = new Set();
    for (const winner of Array.isArray(winners) ? winners : []) {
      const result = imageFile(winner?.image);
      if ('problem' in result) problems.push(`${folder}: ${result.problem}`);
      else if (!files.includes(result.file)) problems.push(`${folder}: no file named ${result.file} in this folder.`);
      else named.add(result.file);
    }
    for (const file of files) {
      if (file === 'challenge.yaml') continue;
      if (!named.has(file)) {
        problems.push(`${folder}/${file}: no winner names this file. Remove it, or fix the winner's \`image\`.`);
        continue;
      }
      if (!IMAGE_EXTENSION.test(file)) problems.push(`${folder}/${file}: export winners' images as WebP or JPEG.`);
      const bytes = statSync(join(directory, folder, file)).size;
      if (bytes > MAX_IMAGE_BYTES) {
        problems.push(
          `${folder}/${file}: ${Math.ceil(bytes / 1000)} KB, over the 500 KB limit. Export it smaller or at a lower ` +
            'quality (see src/content/challenges/README.md).',
        );
      }
    }
  }
  for (const [key, folders] of foldersByWeek) {
    if (folders.length > 1) problems.push(`${key} appears in more than one folder: ${folders.join(', ')}.`);
  }
  return problems;
};

describe('challengeFolderProblems', () => {
  const root = mkdtempSync(join(tmpdir(), 'docs-challenges-'));
  after(() => rmSync(root, { recursive: true, force: true }));
  let count = 0;
  /** @param {Record<string, Record<string, string | number>>} folders file contents, or a size in bytes */
  const directory = (folders) => {
    const path = join(root, String(++count));
    for (const [folder, files] of Object.entries(folders)) {
      mkdirSync(join(path, folder), { recursive: true });
      for (const [name, content] of Object.entries(files)) {
        writeFileSync(join(path, folder, name), typeof content === 'number' ? Buffer.alloc(content) : content);
      }
    }
    return path;
  };
  /** A `challenge.yaml` naming each image as written, in block style. */
  const yaml = (week, start, ...images) =>
    `week: ${week}\ntheme: Test\nstartDate: ${start}\nwinners:\n` +
    images.map((image) => `  - place: honourable\n    artist: A\n    image: ${image}\n`).join('');

  it('accepts folders holding challenge.yaml and the images it names, however the YAML writes the paths', () => {
    const path = directory({
      '2026-w1-a': {
        'challenge.yaml': yaml(1, '2026-01-01', './a.webp', 'b.jpg', '"./c d.jpeg"'),
        'a.webp': 10,
        'b.jpg': MAX_IMAGE_BYTES,
        'c d.jpeg': 10,
        '.DS_Store': 10,
      },
      '2026-w2-b': {
        'challenge.yaml': 'week: 2\nstartDate: 2026-01-08\nwinners: [{ place: 1, artist: A, image: ./e.webp }]\n',
        'e.webp': 10,
      },
    });
    writeFileSync(join(path, 'README.md'), '');
    writeFileSync(join(path, '.DS_Store'), '');
    assert.deepEqual(challengeFolderProblems(path), []);
  });

  it('reports stray files, PNGs, oversized images and folders without challenge.yaml', () => {
    const path = directory({
      '2026-w1-a': {
        'challenge.yaml': yaml(1, '2026-01-01', './big.webp', './art.png'),
        'big.webp': MAX_IMAGE_BYTES + 1,
        'art.png': 10,
        'old.webp': 10,
      },
      '2026-w2-b': { 'challenge.yml': yaml(2, '2026-01-08') },
    });
    writeFileSync(join(path, 'notes.txt'), '');
    const problems = challengeFolderProblems(path).join('\n');
    assert.match(problems, /^2026-w1-a\/big\.webp: 501 KB, over the 500 KB limit/m);
    assert.match(problems, /^2026-w1-a\/art\.png: export winners' images as WebP or JPEG/m);
    assert.match(problems, /^2026-w1-a\/old\.webp: no winner names this file/m);
    assert.match(problems, /^2026-w2-b: has no challenge\.yaml/m);
    assert.match(problems, /^notes\.txt: only challenge folders/m);
  });

  it('reports images that are missing or outside the folder', () => {
    const path = directory({
      '2026-w1-a': { 'challenge.yaml': yaml(1, '2026-01-01', './gone.webp', '../2026-w2-b/x.webp', 'https://x.test/a.webp') },
    });
    assert.deepEqual(challengeFolderProblems(path), [
      '2026-w1-a: no file named gone.webp in this folder.',
      '2026-w1-a: `image: ../2026-w2-b/x.webp` is outside this folder. Put the image beside challenge.yaml.',
      '2026-w1-a: `image: https://x.test/a.webp` must name a file in this folder, such as ./1-artist.webp.',
    ]);
  });

  it('reports YAML that does not parse', () => {
    const path = directory({ '2026-w1-a': { 'challenge.yaml': 'week: [1\n' } });
    assert.match(challengeFolderProblems(path).join('\n'), /^2026-w1-a\/challenge\.yaml: /m);
  });

  it('reports a week used twice in the same year, and allows it in another year', () => {
    const path = directory({
      '2026-w7-a': { 'challenge.yaml': yaml(7, '2026-02-11', './a.webp'), 'a.webp': 10 },
      '2026-w7-b': { 'challenge.yaml': yaml(7, '2026-02-18', './b.webp'), 'b.webp': 10 },
      '2025-w7-c': { 'challenge.yaml': yaml(7, '2025-02-12', './c.webp'), 'c.webp': 10 },
    });
    assert.deepEqual(challengeFolderProblems(path), [
      '2026 week 7 appears in more than one folder: 2026-w7-a, 2026-w7-b.',
    ]);
  });

  it('reports folder names outside <year>-w<week>-<theme>, since they become URLs', () => {
    const path = directory({
      'Week 3': { 'challenge.yaml': yaml(3, '2026-01-14') },
      '2026-w3': { 'challenge.yaml': yaml(3, '2026-01-14') },
    });
    const problems = challengeFolderProblems(path).join('\n');
    assert.match(problems, /^Week 3: name the folder <year>-w<week>-<theme>/m);
    assert.match(problems, /^2026-w3: name the folder/m);
  });

  it("reports a folder whose week or year differs from its challenge.yaml's", () => {
    const path = directory({
      '2026-w45-glass': { 'challenge.yaml': yaml(44, '2026-10-28') },
      '2027-w1-frost': { 'challenge.yaml': yaml(1, '2026-12-30') },
    });
    assert.deepEqual(challengeFolderProblems(path), [
      '2026-w45-glass: the folder says week 45, but challenge.yaml says `week: 44`.',
      '2027-w1-frost: the folder says 2027, but the challenge started in 2026.',
    ]);
  });
});

describe('src/content/challenges', () => {
  it('holds only well-formed challenge folders', () => {
    assert.deepEqual(challengeFolderProblems(fileURLToPath(new URL('../content/challenges/', import.meta.url))), []);
  });
});

// Weekly Discord art challenges: one folder per week in `src/content/challenges/`, shown on `/showcase/`, on
// `/showcase/challenges/` and on the home page. Challenge content and these labels stay English: the showcase pages
// are root-only, and the home page takes its own labels from `en.json` and formats dates in the page's language.

/** A winner's `place`, in display order: `challenge.yaml` lists its winners in this order, ties side by side. */
export const PLACES = /** @type {const} */ ([1, 2, 3, 'runner-up', 'honourable']);

/** @typedef {(typeof PLACES)[number]} Place */

/** @type {Record<Place, string>} */
const PLACE_LABELS = {
  1: '1st place',
  2: '2nd place',
  3: '3rd place',
  'runner-up': 'Runner-up',
  honourable: 'Honourable mention',
};

/** @param {Place} place */
export const placeLabel = (place) => PLACE_LABELS[place];

/**
 * Newest first: by end date, then by week, then by id, so builds are stable.
 * @template {{ id: string, data: { week?: number, endDate: Date } }} T
 * @param {readonly T[]} challenges
 * @returns {T[]}
 */
export const sortChallenges = (challenges) =>
  [...challenges].sort(
    (a, b) =>
      b.data.endDate.getTime() - a.data.endDate.getTime() ||
      (b.data.week ?? 0) - (a.data.week ?? 0) ||
      a.id.localeCompare(b.id),
  );

/** At most this many winners hold a numbered place, ties included; the podium has room for no more. */
const MAX_NUMBERED = 3;

/**
 * What is wrong with a challenge's `winners`, one message per problem, for the schema. The pages show winners in file
 * order, so the file lists them in `PLACES` order: numbered places first and never decreasing (a tie repeats a place,
 * such as 1, 2, 2), then runners-up, then honourable mentions. At least one winner holds a numbered place: it leads the
 * page and is its preview image.
 * @param {readonly { place: Place, artist: string }[]} winners
 * @returns {string[]}
 */
export const placeProblems = (winners) => {
  const problems = [];
  const numbered = winners.filter(({ place }) => typeof place === 'number').length;
  if (numbered === 0) {
    problems.push('No winner has a numbered place. Give the top winner `place: 1` (or 2 or 3), as announced.');
  }
  if (numbered > MAX_NUMBERED) {
    problems.push(
      `${numbered} winners have a numbered place; at most ${MAX_NUMBERED} can. Use \`runner-up\` or \`honourable\` ` +
        'for the rest.',
    );
  }
  winners.forEach((winner, index) => {
    const previous = winners[index - 1];
    if (previous && PLACES.indexOf(winner.place) < PLACES.indexOf(previous.place)) {
      problems.push(
        `Winner ${index + 1} (${winner.artist}, \`place: ${winner.place}\`) comes after ` +
          `\`place: ${previous.place}\`. List winners in the order the page shows them: 1, 2, 3, runner-up, ` +
          'honourable, with ties side by side.',
      );
    }
  });
  return problems;
};

/**
 * The challenges either side of `id` in `sorted` (newest first): `older` is the week before, `newer` the week after.
 * @template {{ id: string }} T
 * @param {readonly T[]} sorted
 * @param {string} id
 * @returns {{ older: T | undefined, newer: T | undefined }}
 */
export const adjacentChallenges = (sorted, id) => {
  const index = sorted.findIndex((challenge) => challenge.id === id);
  if (index < 0) throw new Error(`No challenge has the id "${id}".`);
  return { older: sorted[index + 1], newer: sorted[index - 1] };
};

/**
 * Alt text for a winner's image: "<title> by <artist>", or for an untitled entry the theme and the place, such as
 * "Lost, runner-up by Romeo".
 * @param {{ title?: string, artist: string, place: Place }} winner
 * @param {string} theme
 */
export const winnerAlt = ({ title, artist, place }, theme) =>
  `${title ?? `${theme}, ${placeLabel(place).toLowerCase()}`} by ${artist}`;

/** Frame shapes stay within these aspect ratios (width / height); a more extreme image shows whole inside its frame. */
const MIN_ASPECT = 0.5;
const MAX_ASPECT = 2.2;
/** A lower place's frame is at most this fraction of the lead's height, and at most this fraction of its area. */
const SIDE_HEIGHT = 0.7;
const SIDE_AREA = 0.6;
/** Runners-up and honourable mentions are at most this fraction of the lead's height, and shorter than any side. */
const MENTION_HEIGHT = 0.55;

/**
 * A frame's aspect ratio: the image's own, within limits.
 * @param {{ width: number, height: number }} image
 */
export const frameAspect = ({ width, height }) => Math.min(MAX_ASPECT, Math.max(MIN_ASPECT, width / height));

/** @param {number} value */
const round = (value) => Math.round(value * 1e4) / 1e4;

/**
 * The podium's frames on wide screens: the numbered winners in one row, bottoms aligned, each frame its image's shape
 * (`aspect`) and `scale` times the lead's height. Tied winners share a height, so they read as equal, at their own
 * widths. A lower place is shorter than the lead and smaller in area, whatever its shape; tied lower places take the
 * height that keeps the widest of them so. `span` is the row's width in lead heights, gaps aside, and `mentionScale`
 * the most the runners-up' and honourable mentions' height may be, in lead heights.
 * @param {readonly { place: Place, image: { width: number, height: number } }[]} podium the numbered winners, in order
 * @returns {{ frames: { aspect: number, scale: number }[], span: number, mentionScale: number }}
 */
export const podiumLayout = (podium) => {
  const aspects = podium.map(({ image }) => frameAspect(image));
  const leadPlace = podium[0]?.place;
  /** @type {Map<Place, number>} the widest frame of each lower place */
  const widest = new Map();
  podium.forEach(({ place }, index) => {
    if (place !== leadPlace) widest.set(place, Math.max(widest.get(place) ?? 0, aspects[index]));
  });
  const frames = podium.map(({ place }, index) => {
    const widestTied = widest.get(place);
    const scale =
      widestTied === undefined ? 1 : Math.min(SIDE_HEIGHT, Math.sqrt((SIDE_AREA * aspects[0]) / widestTied));
    return { aspect: round(aspects[index]), scale: round(scale) };
  });
  const sideScales = frames.filter(({ scale }) => scale < 1).map(({ scale }) => scale);
  return {
    frames,
    span: round(frames.reduce((sum, { aspect, scale }) => sum + aspect * scale, 0)),
    mentionScale: round(Math.min(MENTION_HEIGHT, 0.9 * Math.min(...sideScales))),
  };
};

/**
 * For frames shown two to a row at one height per row, as on phones: each frame's row span, the sum of its row's
 * aspect ratios, so a frame is `aspect / span` of the row's width. A last frame alone keeps the row above's height,
 * or with no row above, the height a square beside it would give.
 * @param {readonly number[]} aspects
 * @returns {number[]}
 */
export const pairSpans = (aspects) =>
  aspects.map((aspect, index) => {
    const start = index - (index % 2);
    if (start + 1 < aspects.length) return round(aspects[start] + aspects[start + 1]);
    return round(start > 0 ? aspects[start - 2] + aspects[start - 1] : aspect + 1);
  });

/**
 * A challenge's dates in the page's language, such as "October 21 – 27, 2026" in English.
 * @param {Date} start
 * @param {Date} end
 * @param {string} [lang] a BCP 47 language tag; English by default
 */
export const formatChallengeDates = (start, end, lang = 'en') =>
  // Challenge dates are calendar days, which YAML gives as midnight UTC; another time zone could show the day before.
  new Intl.DateTimeFormat(lang, { dateStyle: 'long', timeZone: 'UTC' }).formatRange(start, end);

/** The archive of every challenge, a root-only page. */
export const CHALLENGES_PATH = '/showcase/challenges/';

/** @param {string} id a challenge's folder name */
export const challengePath = (id) => `${CHALLENGES_PATH}${id}/`;

/** Where the challenges are announced and entered. */
export const DISCORD_INVITE_URL = 'https://discord.gg/ZmtBAhwWhy';

// Weekly Discord art challenges: one folder per week in `src/content/challenges/`, shown on `/showcase/`, on
// `/showcase/challenges/` and on the home page. Challenge content and these labels stay English: the showcase pages
// are root-only, and the home page takes its own labels from `en.json`.

/** A winner's `place`, in display order. Places 1 to 3 go to one winner each; any number of honourable mentions. */
export const PLACES = /** @type {const} */ ([1, 2, 3, 'honourable']);

/** @typedef {(typeof PLACES)[number]} Place */

/** @type {Record<Place, string>} */
const PLACE_LABELS = { 1: '1st place', 2: '2nd place', 3: '3rd place', honourable: 'Honourable mention' };

/** @param {Place} place */
export const placeLabel = (place) => PLACE_LABELS[place];

/**
 * Newest first: by end date, then by week, then by id, so builds are stable.
 * @template {{ id: string, data: { week: number, endDate: Date } }} T
 * @param {readonly T[]} challenges
 * @returns {T[]}
 */
export const sortChallenges = (challenges) =>
  [...challenges].sort(
    (a, b) =>
      b.data.endDate.getTime() - a.data.endDate.getTime() || b.data.week - a.data.week || a.id.localeCompare(b.id),
  );

/**
 * 1st, 2nd, 3rd, then honourable mentions in the order the file lists them.
 * @template {{ place: Place }} W
 * @param {readonly W[]} winners
 * @returns {W[]}
 */
export const orderWinners = (winners) =>
  [...winners].sort((a, b) => PLACES.indexOf(a.place) - PLACES.indexOf(b.place));

/**
 * The places 1 to 3 that more than one winner holds, for the schema's error message.
 * @param {readonly { place: Place }[]} winners
 * @returns {Place[]}
 */
export const repeatedPlaces = (winners) =>
  PLACES.filter((place) => place !== 'honourable' && winners.filter((winner) => winner.place === place).length > 1);

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
 * Alt text for a winner's image: "<title> by <artist>", or "Week <week> winner by <artist>" for an untitled entry.
 * @param {{ title?: string, artist: string }} winner
 * @param {number} week
 */
export const winnerAlt = ({ title, artist }, week) => `${title ?? `Week ${week} winner`} by ${artist}`;

// Challenge dates are calendar days, which YAML gives as midnight UTC; any other time zone could show the day before.
const dateFormat = new Intl.DateTimeFormat('en', { dateStyle: 'long', timeZone: 'UTC' });

/**
 * A challenge's dates, such as "October 21 – 27, 2026".
 * @param {Date} start
 * @param {Date} end
 */
export const formatChallengeDates = (start, end) => dateFormat.formatRange(start, end);

/** The archive of every challenge, a root-only page. */
export const CHALLENGES_PATH = '/showcase/challenges/';

/** @param {string} id a challenge's folder name */
export const challengePath = (id) => `${CHALLENGES_PATH}${id}/`;

/** Where the challenges are announced and entered. */
export const DISCORD_INVITE_URL = 'https://discord.gg/ZmtBAhwWhy';

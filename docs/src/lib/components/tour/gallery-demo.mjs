// The Gallery tour demo's sample data and the pure rules it shows them by, after the app's gallery
// (`invokeai/frontend/webv2/src/features/gallery/ui/galleryBoardGroups.ts`, `GalleryBoardsPanel.tsx` and
// `GalleryWidgetView.tsx`). `GalleryDemo.astro` renders the initial state from these and its script applies them on
// every change, so the page without JavaScript and the page with it agree.

/** @typedef {'board' | 'uncategorized'} GalleryDemoBoardKind */
/**
 * @typedef {object} GalleryDemoBoard
 * @property {string} id
 * @property {string} name
 * @property {GalleryDemoBoardKind} kind
 * @property {string | null} projectId
 * @property {boolean} [isInbox]
 * @property {string | null} cover the image id of its cover, or none for an icon
 */
/** @typedef {'images' | 'assets'} GalleryDemoView */
/**
 * @typedef {object} GalleryDemoItem
 * @property {string} id
 * @property {string} boardId
 * @property {GalleryDemoView} view `images` lists under Media, `assets` under Uploads, as in the app
 * @property {string} image the image id
 * @property {string} name
 * @property {string} prompt what the item search matches besides the name
 * @property {number} width
 * @property {number} height
 * @property {number} createdAt higher is newer
 * @property {boolean} starred
 */

export const GALLERY_DEMO_PROJECT = { id: 'coastal-series', name: 'Coastal Series' };

/** @type {readonly GalleryDemoBoard[]} */
export const GALLERY_DEMO_BOARDS = [
  { id: 'inbox', name: 'Inbox', kind: 'board', projectId: GALLERY_DEMO_PROJECT.id, isInbox: true, cover: null },
  { id: 'concept-art', name: 'Concept Art', kind: 'board', projectId: GALLERY_DEMO_PROJECT.id, cover: 'car' },
  { id: 'portraits', name: 'Portraits', kind: 'board', projectId: GALLERY_DEMO_PROJECT.id, cover: 'portrait' },
  { id: 'landscapes', name: 'Landscapes', kind: 'board', projectId: GALLERY_DEMO_PROJECT.id, cover: 'mountains' },
  { id: 'uncategorized', name: 'Uncategorized', kind: 'uncategorized', projectId: null, cover: null },
  { id: 'architecture', name: 'Architecture', kind: 'board', projectId: null, cover: 'house' },
  { id: 'product-shots', name: 'Product Shots', kind: 'board', projectId: null, cover: 'espresso' },
];

export const GALLERY_DEMO_INITIAL_BOARD_ID = 'concept-art';

/** @param {string} id @param {string} boardId @param {GalleryDemoView} view @param {string} image */
const item = (id, boardId, view, image, name, prompt, createdAt, starred = false, width = 1024, height = 1024) => ({
  id,
  boardId,
  view,
  image,
  name,
  prompt,
  width,
  height,
  createdAt,
  starred,
});

/** @type {readonly GalleryDemoItem[]} */
export const GALLERY_DEMO_ITEMS = [
  item('i1', 'inbox', 'images', 'rider', 'Night courier', 'masked rider neon city night', 30),
  item('i2', 'inbox', 'images', 'scientist', 'Lab sketch', 'cartoon scientist flasks experiment', 29),

  item('c1', 'concept-art', 'images', 'car', 'Copper coupe', 'orange concept car street', 28, false, 1216, 832),
  item('c2', 'concept-art', 'images', 'mech', 'Brass sentinel', 'gold mech robot city', 27, true),
  item('c3', 'concept-art', 'images', 'wizard', 'Torchbearer', 'goblin wizard holding a torch', 26, true, 832, 1216),
  item('c4', 'concept-art', 'images', 'rider', 'Courier helmet study', 'masked rider neon', 25),
  item('c5', 'concept-art', 'images', 'planet', 'Ember planet', 'impasto painting planet sky', 24),
  item('c6', 'concept-art', 'assets', 'lineart', 'Jacket sketch', 'line art man jacket reference', 23, false, 640, 640),

  item('p1', 'portraits', 'images', 'portrait', 'Studio smile', 'portrait woman smiling studio', 22, true),
  item('p2', 'portraits', 'images', 'lineart', 'Ink portrait', 'line art man jacket', 21, false, 640, 640),
  item('p3', 'portraits', 'images', 'scientist', 'Professor', 'cartoon scientist flasks', 20),
  item('p4', 'portraits', 'images', 'wizard', 'Goblin mage', 'goblin wizard torch', 19, false, 832, 1216),

  item('l1', 'landscapes', 'images', 'mountains', 'Alpine meadow', 'mountains meadow sunset rock', 18, true),
  item('l2', 'landscapes', 'images', 'planet', 'Red giant', 'impasto painting planet', 17),
  item('l3', 'landscapes', 'images', 'market', 'Harbor market', 'market stalls fruit crowd', 16),

  item('u1', 'uncategorized', 'images', 'market', 'Market morning', 'market stalls fruit', 15),
  item('u2', 'uncategorized', 'assets', 'portrait', 'Headshot reference', 'portrait woman', 14),
  item('u3', 'uncategorized', 'assets', 'lineart', 'Outline reference', 'line art man', 13, false, 640, 640),

  item('a1', 'architecture', 'images', 'house', 'Cedar house', 'modern wooden house autumn', 12, true, 1216, 832),
  item('a2', 'architecture', 'images', 'mountains', 'Site survey', 'mountains meadow', 11),

  item('s1', 'product-shots', 'images', 'espresso', 'Espresso machine', 'coffee machine product shot', 10, true),
  item('s2', 'product-shots', 'images', 'sushi', 'Sushi plate', 'sushi plate salmon roe', 9),
  item('s3', 'product-shots', 'images', 'car', 'Coupe, three quarter', 'orange concept car', 8, false, 1216, 832),
  item('s4', 'product-shots', 'assets', 'espresso', 'Machine reference', 'coffee machine photo', 7),
];

/** The name a board row shows. */
export const getBoardLabel = (/** @type {GalleryDemoBoard} */ board) => board.name;

/**
 * The board list's two tiers, as the app groups them: the open project's boards with its inbox first, then the
 * Library with Uncategorized first. A search keeps the boards whose name contains it, and offers to create a board
 * when no board on screen has exactly that name.
 * @param {readonly GalleryDemoBoard[]} boards
 * @param {string} projectId
 * @param {string} searchTerm
 */
export const getBoardGroups = (boards, projectId, searchTerm) => {
  const term = searchTerm.trim().toLowerCase();
  const matches = (/** @type {GalleryDemoBoard} */ board) => !term || getBoardLabel(board).toLowerCase().includes(term);
  const projectBoards = boards.filter((board) => board.kind === 'board' && board.projectId === projectId && matches(board));
  projectBoards.sort((left, right) => Number(Boolean(right.isInbox)) - Number(Boolean(left.isInbox)));
  const libraryBoards = [
    ...boards.filter((board) => board.kind === 'uncategorized' && matches(board)),
    ...boards.filter((board) => board.kind === 'board' && board.projectId === null && matches(board)),
  ];
  const shown = [...projectBoards, ...libraryBoards];
  // Each tier counts its boards, the inbox among them; Uncategorized is where media on no board shows, not a board.
  const countBoards = (/** @type {GalleryDemoBoard[]} */ list) => list.filter((board) => board.kind === 'board').length;

  return {
    projectBoards,
    libraryBoards,
    projectCount: countBoards(projectBoards),
    libraryCount: countBoards(libraryBoards),
    hasAnyMatch: shown.length > 0,
    canCreateFromSearch: term.length > 0 && !shown.some((board) => getBoardLabel(board).toLowerCase() === term),
  };
};

/**
 * A board's media and uploads, as its count badge (`media | uploads`) and the view tabs show them.
 * @param {readonly GalleryDemoItem[]} items
 * @param {string} boardId
 */
export const getBoardCounts = (items, boardId) => {
  let media = 0;
  let uploads = 0;
  for (const entry of items) {
    if (entry.boardId !== boardId) continue;
    if (entry.view === 'images') media += 1;
    else uploads += 1;
  }
  return { media, uploads };
};

/**
 * What the grid shows: the listing for the board, view and search in the chosen order, and the pinned Starred strip
 * above it. With the starred filter on, the listing holds only starred items and there is no strip, which would
 * repeat it.
 * @param {readonly GalleryDemoItem[]} items
 * @param {{ boardId: string, view: GalleryDemoView, starredOnly: boolean, searchTerm: string, order: 'DESC' | 'ASC' }} filter
 */
export const getGridSections = (items, { boardId, view, starredOnly, searchTerm, order }) => {
  const term = searchTerm.trim().toLowerCase();
  const matches = (/** @type {GalleryDemoItem} */ entry) =>
    !term || `${entry.name} ${entry.prompt}`.toLowerCase().includes(term);
  const listed = items
    .filter((entry) => entry.boardId === boardId && entry.view === view && matches(entry))
    .sort((left, right) => (order === 'DESC' ? right.createdAt - left.createdAt : left.createdAt - right.createdAt));
  const starred = listed.filter((entry) => entry.starred);

  return starredOnly ? { starred: [], listing: starred } : { starred, listing: listed };
};

/**
 * The grid's empty state: `upload` for a board with nothing in the view, or the message the app shows when a filter
 * or search hides everything. Null when the listing has items.
 * @param {number} listingCount
 * @param {{ starredOnly: boolean, searchTerm: string }} filter
 */
export const getGridEmptyState = (listingCount, { starredOnly, searchTerm }) => {
  if (listingCount > 0) return null;
  if (starredOnly) return { kind: 'message', text: 'No starred items match this board, tab, or search.' };
  if (searchTerm.trim()) return { kind: 'message', text: 'No items match this board, tab, or search.' };
  return { kind: 'upload', text: 'Drop media here or click to upload' };
};

/**
 * The boards with a new, empty one named `name` in the open project or the Library, and the new board's id.
 * @param {readonly GalleryDemoBoard[]} boards
 * @param {string} name
 * @param {string | null} projectId
 */
export const createBoard = (boards, name, projectId) => {
  const id = `created-${boards.length}`;
  /** @type {GalleryDemoBoard} */
  const board = { id, name: name.trim(), kind: 'board', projectId, cover: null };
  return { boards: [...boards, board], board };
};

/**
 * The items with one item's star flipped.
 * @param {readonly GalleryDemoItem[]} items
 * @param {string} itemId
 */
export const toggleStarred = (items, itemId) =>
  items.map((entry) => (entry.id === itemId ? { ...entry, starred: !entry.starred } : entry));

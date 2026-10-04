/** Horizontal inset of rows and headers from the list's edge. */
export const LIST_ROW_INSET = '0.75rem';

/** Start padding of the row checkbox inside the row surface; matches the `check` slot of the list item recipe. */
const LIST_CHECK_PADDING = '0.5rem';

/** Where row checkboxes start relative to the list's edge, so chrome above the list can line up with them. */
export const LIST_CHECK_INSET = `calc(${LIST_ROW_INSET} + ${LIST_CHECK_PADDING})`;

/** A narrow gap between rows; a divider, when shown, sits centred in it. */
export const LIST_ROW_GAP_PX = 2;

/** Dividers stop short of the row edges so a filled row's rounded corners never meet the line. */
export const LIST_DIVIDER_INSET = '0.5rem';

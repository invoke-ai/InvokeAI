/**
 * Wraps each table in a `div.table-scroll`. The wrapper carries the table's border and scrolls a table wider than the
 * content column, so the table itself can fill the column with ordinary table layout and size its columns to their
 * longest word or code token instead of breaking them. The wrapper's `data-label` names the table, from its caption
 * or else the nearest heading before it, for the scroll region the page makes of a table that overflows
 * (`src/lib/components/MarkdownContent.astro`).
 */
export function rehypeTableScroll() {
  return (tree) => {
    let lastHeading = '';
    const visit = (node) => {
      if (!Array.isArray(node.children)) return;
      node.children = node.children.map((child) => {
        if (child.type !== 'element') return child;
        if (/^h[1-6]$/.test(child.tagName)) lastHeading = textOf(child).trim();
        if (child.tagName !== 'table') {
          visit(child);
          return child;
        }
        const caption = child.children.find((cell) => cell.type === 'element' && cell.tagName === 'caption');
        const label = (caption && textOf(caption).trim()) || lastHeading || 'Table';
        return { type: 'element', tagName: 'div', properties: { className: ['table-scroll'], dataLabel: label }, children: [child] };
      });
    };
    visit(tree);
  };
}

const textOf = (node) =>
  node.type === 'text' ? node.value : Array.isArray(node.children) ? node.children.map(textOf).join('') : '';

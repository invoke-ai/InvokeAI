import { describe, expect, it } from 'vitest';

import { getWorkflowContextMenuState } from './workflowContextMenu';

class FakeElement {
  readonly children: FakeElement[] = [];
  readonly dataset: Record<string, string> = {};
  parent: FakeElement | null = null;

  constructor(
    readonly tagName: string,
    readonly options: { className?: string; attrs?: Record<string, string>; dataset?: Record<string, string> } = {}
  ) {
    if (options.dataset) {
      Object.assign(this.dataset, options.dataset);
    }
  }

  appendChild(child: FakeElement): FakeElement {
    child.parent = this;
    this.children.push(child);
    return child;
  }

  contains(target: FakeElement): boolean {
    return target === this || this.children.some((child) => child.contains(target));
  }

  closest(selector: string): FakeElement | null {
    const selectors = selector.split(',').map((s) => s.trim());
    let current: FakeElement | null = this.parent;
    if (selectors.some((selector) => this.matches(selector))) {
      return this;
    }
    while (current) {
      if (selectors.some((selector) => current?.matches(selector))) {
        return current;
      }
      current = current.parent;
    }
    return null;
  }

  private matches(selector: string): boolean {
    if (selector.startsWith('.')) {
      return this.options.className?.split(' ').includes(selector.slice(1)) ?? false;
    }

    if (selector === '[data-connector-node-id]') {
      return typeof this.dataset.connectorNodeId === 'string';
    }

    if (selector === 'a[href]') {
      return this.tagName === 'a' && typeof this.options.attrs?.href === 'string';
    }

    if (selector === '[contenteditable="true"]') {
      return this.options.attrs?.contenteditable === 'true';
    }

    if (selector === '[contenteditable=""]') {
      return this.options.attrs?.contenteditable === '';
    }

    if (selector === '[role="textbox"]') {
      return this.options.attrs?.role === 'textbox';
    }

    return this.tagName === selector;
  }
}

const buildEvent = (target: FakeElement, shiftKey = false) =>
  ({
    shiftKey,
    target,
    clientX: 10,
    clientY: 20,
    pageX: 30,
    pageY: 40,
  }) as unknown as MouseEvent;

const buildFlow = () => {
  const wrapper = new FakeElement('div');
  const pane = wrapper.appendChild(new FakeElement('div', { className: 'react-flow__pane' }));
  return { wrapper, pane };
};

describe('getWorkflowContextMenuState', () => {
  it('opens the add connector menu for pane descendants', () => {
    const { wrapper, pane } = buildFlow();
    const background = pane.appendChild(new FakeElement('svg'));

    expect(getWorkflowContextMenuState(buildEvent(background), wrapper as unknown as HTMLDivElement)).toEqual({
      kind: 'pane',
      clientX: 10,
      clientY: 20,
      pageX: 30,
      pageY: 40,
    });
  });

  it('opens the connector menu for connector nodes even though they are inside a ReactFlow node', () => {
    const { wrapper, pane } = buildFlow();
    const node = pane.appendChild(new FakeElement('div', { className: 'react-flow__node' }));
    const connector = node.appendChild(
      new FakeElement('div', { dataset: { connectorNodeId: 'connector-1' }, className: 'nodrag' })
    );

    expect(getWorkflowContextMenuState(buildEvent(connector), wrapper as unknown as HTMLDivElement)).toEqual({
      kind: 'connector',
      connectorId: 'connector-1',
      pageX: 30,
      pageY: 40,
    });
  });

  it('does not open the add connector menu for interactive fields inside a node', () => {
    const { wrapper, pane } = buildFlow();
    const node = pane.appendChild(new FakeElement('div', { className: 'react-flow__node' }));
    const input = node.appendChild(new FakeElement('input'));

    expect(getWorkflowContextMenuState(buildEvent(input), wrapper as unknown as HTMLDivElement)).toBeNull();
  });

  it('does not open the add connector menu for image field areas inside a node', () => {
    const { wrapper, pane } = buildFlow();
    const node = pane.appendChild(new FakeElement('div', { className: 'react-flow__node' }));
    const imageField = node.appendChild(new FakeElement('div', { className: 'nodrag' }));

    expect(getWorkflowContextMenuState(buildEvent(imageField), wrapper as unknown as HTMLDivElement)).toBeNull();
  });

  it('allows the native context menu on shift right click', () => {
    const { wrapper, pane } = buildFlow();

    expect(getWorkflowContextMenuState(buildEvent(pane, true), wrapper as unknown as HTMLDivElement)).toBeNull();
  });
});

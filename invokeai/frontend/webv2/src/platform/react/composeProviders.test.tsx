import type { ReactNode } from 'react';

import { createContext, use } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { composeProviders } from './composeProviders';

// Three probe contexts. Each provider chains off the previous context's value,
// so the probe output encodes the exact nesting order.
const FirstContext = createContext('missing-first');
const SecondContext = createContext('missing-second');
const ThirdContext = createContext('missing-third');

const FirstProvider = ({ children }: { children: ReactNode }) => (
  <FirstContext.Provider value="first">{children}</FirstContext.Provider>
);

const SecondProvider = ({ children }: { children: ReactNode }) => (
  <SecondContext.Provider value={`${use(FirstContext)}/second`}>{children}</SecondContext.Provider>
);

const ThirdProvider = ({ children }: { children: ReactNode }) => (
  <ThirdContext.Provider value={`${use(SecondContext)}/third`}>{children}</ThirdContext.Provider>
);

const Probe = () => <output>{use(ThirdContext)}</output>;

describe('composeProviders', () => {
  it('nests providers in array order, outermost first', () => {
    const Composed = composeProviders([FirstProvider, SecondProvider, ThirdProvider]);

    const markup = renderToStaticMarkup(
      <Composed>
        <Probe />
      </Composed>
    );

    expect(markup).toContain('first/second/third');
  });

  it('renders children unchanged for an empty provider list', () => {
    const Composed = composeProviders([]);

    const markup = renderToStaticMarkup(<Composed>plain children</Composed>);

    expect(markup).toBe('plain children');
  });

  it('returns one stable component identity per call, so a module-scope composition is render-stable', () => {
    const Composed = composeProviders([FirstProvider, SecondProvider, ThirdProvider]);

    // The single composed reference renders identically every time it is used.
    const renderOnce = () =>
      renderToStaticMarkup(
        <Composed>
          <Probe />
        </Composed>
      );
    expect(renderOnce()).toBe(renderOnce());

    // A second call creates a distinct component: composing inside render would
    // change element type each render and remount the subtree. Module scope only.
    expect(composeProviders([FirstProvider, SecondProvider, ThirdProvider])).not.toBe(Composed);
  });
});

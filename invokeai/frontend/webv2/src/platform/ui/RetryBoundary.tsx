import type { ReactNode } from 'react';

import { Stack, Text } from '@chakra-ui/react';
import { Component } from 'react';

import { Button } from './Button';

/** Isolates a deferred editor and lets its owner reset a failed code resource. */
export class RetryBoundary extends Component<
  { children: ReactNode; retry: () => Promise<unknown>; message: string; retryLabel: string },
  { failed: boolean; retrying: boolean }
> {
  state = { failed: false, retrying: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  handleRetry = () => {
    if (this.state.retrying) {
      return;
    }
    this.setState({ retrying: true });
    // Enter through a promise so synchronous loader errors follow the same recovery path.
    void Promise.resolve()
      .then(() => this.props.retry())
      .then(
        () => this.setState({ failed: false, retrying: false }),
        () => this.setState({ retrying: false })
      );
  };
  render() {
    if (!this.state.failed) {
      return this.props.children;
    }
    return (
      <Stack role="alert" gap="2" aria-busy={this.state.retrying}>
        <Text fontSize="xs" color="fg.error">
          {this.props.message}
        </Text>
        <Button size="xs" loading={this.state.retrying} onClick={this.handleRetry}>
          {this.props.retryLabel}
        </Button>
      </Stack>
    );
  }
}

export interface NodeInvocationStartedEvent {
  invocation_source_id: string;
}

export interface NodeInvocationCompleteEvent {
  invocation_source_id: string;
  result: unknown;
}

export interface NodeInvocationErrorEvent {
  invocation_source_id: string;
  error_message: string;
}

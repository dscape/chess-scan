import { Component, type ErrorInfo, type ReactNode } from "react";

type AppErrorBoundaryProps = {
  children: ReactNode;
};

type AppErrorBoundaryState = {
  error: Error | null;
};

export default class AppErrorBoundary extends Component<
  AppErrorBoundaryProps,
  AppErrorBoundaryState
> {
  state: AppErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): AppErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Chess Scan could not render the current page", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <main className="route-loading" role="alert">
        <div className="route-loading__error">
          <h1>Something went wrong.</h1>
          <p>{this.state.error.message || "The page could not be displayed."}</p>
          <button
            type="button"
            className="primary-button"
            onClick={() => window.location.replace("/")}
          >
            Return to the scanner
          </button>
        </div>
      </main>
    );
  }
}

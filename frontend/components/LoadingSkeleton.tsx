interface LoadingSkeletonProps {
  className?: string;
}

export function LoadingSkeleton({ className = "" }: LoadingSkeletonProps) {
  return (
    <div className={`skeleton rounded-lg ${className}`} />
  );
}

export function ProductCardSkeleton() {
  return (
    <div className="flex flex-col gap-3 p-4 rounded-xl border border-bg-border bg-bg-surface">
      <div className="flex items-start gap-3">
        <LoadingSkeleton className="w-10 h-10 rounded-lg shrink-0" />
        <div className="flex-1 flex flex-col gap-2">
          <LoadingSkeleton className="h-4 w-3/4" />
          <LoadingSkeleton className="h-3 w-1/2" />
        </div>
        <LoadingSkeleton className="w-12 h-5 rounded-full shrink-0" />
      </div>
      <LoadingSkeleton className="h-1 w-full rounded-full" />
      <LoadingSkeleton className="h-3 w-1/3" />
    </div>
  );
}

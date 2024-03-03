import PoseDetection from "./_components/PoseDetection";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col gap-8 items-center p-8 md:p-16 lg:p-24">
      <div>
        <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl">
          POSE-MAX
        </h1>
      </div>

      <div>
        <PoseDetection />
      </div>
    </main>
  );
}

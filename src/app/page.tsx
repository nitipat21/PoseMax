import PoseDetection from "./_components/PoseDetection";

export default function Home() {
  return (
    <main className="relative">
      <div
        id="bg"
        className="w-full min-h-screen h-full flex -z-10 absolute top-0 left-0"
      ></div>
      <div className="flex min-h-screen flex-col gap-8 items-center p-8 md:p-16 lg:p-24 z-10 relative">
        <div>
          <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl">
            POSE-MAX
          </h1>
        </div>

        <div className="flex items-center flex-col">
          <PoseDetection />
        </div>
      </div>
    </main>
  );
}

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { fetchQuery } from "@/lib/http";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@radix-ui/react-popover";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";

export function SearchPage() {
  const [query, setQuery] = useState<string>("");
  const [nprobe, setNprobe] = useState<number>(5);
  const [nresults, setNresults] = useState<number>(5);
  const [images, setImages] = useState<string[]>([]);

  const onSearch = async (_: React.MouseEvent<HTMLButtonElement>) => {
    const images = await fetchQuery(query, nprobe, nresults);
    setImages(images);
  };

  return (
    <div className="flex flex-col items-center justify-start h-screen w-screen">
      <h1 className="text-3xl mt-8"> Search engine </h1>
      <div className="w-7/12 mt-16 flex flex-row items-center justify-center gap-4">
        <Input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Button onClick={onSearch}> Search </Button>
      </div>
      <div className="flex items-center justify-center mt-4">
        <Popover>
          <PopoverTrigger>
            <Button variant={"secondary"}>Advanced Settings</Button>
          </PopoverTrigger>
          <PopoverContent>
            <Card>
              <CardHeader>
                <CardTitle> Advanced Search Settings </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-col items-center gap-2">
                  <Label htmlFor="nresults"> Number of search results </Label>
                  <Input
                    value={nresults}
                    type="number"
                    onChange={(e) => setNresults(parseInt(e.target.value))}
                    id="nresults"
                  />
                </div>
                <div className="flex flex-col items-center gap-2 mt-4">
                  <Label htmlFor="nprobe">
                    Number of searched clusters (big number = more accurate
                    results, slower search)
                  </Label>
                  <Input
                    value={nprobe}
                    type="number"
                    onChange={(e) => setNprobe(parseInt(e.target.value))}
                    id="nprobe"
                  />
                </div>
              </CardContent>
            </Card>
          </PopoverContent>
        </Popover>
      </div>
      <div className="grid grid-cols-3 w-10/12 gap-4 mt-8 overflow-scroll pb-8">
        {images.map((image, idx) => {
          return (
            <div className="flex items-center justify-center bg-gray-100 rounded-xl border-sky-50 p-4">
              <img
                src={`data:image/jpeg;base64,${image}`}
                key={idx}
                className="max-h-100"
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

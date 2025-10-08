const URL = "http://127.0.0.1:2137/";

interface QueryDataDto {
  query: string;
  nprobe: number;
  nresults: number;
}

interface QueryResponseDataDto {
  images: string[];
}

export const fetchQuery = async (
  query: string,
  nprobe: number,
  nresults: number,
): Promise<string[]> => {
  const reqBody: QueryDataDto = { query, nprobe, nresults };
  const res = await fetch(URL + "query", {
    method: "POST",
    body: JSON.stringify(reqBody),
    headers: {
      "Content-type": "application/json",
    },
  });
  if (!res.ok) {
    throw new Error("error fetching query results from server");
  }
  const body = (await res.json()) as QueryResponseDataDto;
  return body.images;
};

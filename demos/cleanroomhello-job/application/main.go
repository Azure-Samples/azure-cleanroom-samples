package main

import (
   "compress/gzip"
   "fmt"
   "io"
   "os"
   "time"
)

func waitForShareToBeReady(filePath string) {

   maxCounter := 300
   for true {
      if _, err := os.Stat(filePath); err == nil {
         fmt.Println("File is present at:", filePath)
         break
      }  else if os.IsNotExist(err) {
            fmt.Println("File is not yet present at:", filePath)
            if maxCounter <= 0 {
               fmt.Println("Timeout waiting for file at:", filePath)
               panic(err)
            }
            maxCounter--
            time.Sleep(2 * time.Second)
      }
   }
}

func main() {

   inputLocation := os.Getenv("INPUT_LOCATION")
   filePath := inputLocation + "/input.txt"

   waitForShareToBeReady(filePath)

   fmt.Println("Opening the input file.")
   inputFile, err := os.Open(filePath)
   if err != nil {
      panic(err)
   }
   defer inputFile.Close()

   fmt.Println("Creating the output file.")
   outputLocation := os.Getenv("OUTPUT_LOCATION")
   outputFile, err := os.Create(outputLocation + "/output.gz")
   if err != nil {
      panic(err)
   }
   defer outputFile.Close()

   gzipWriter := gzip.NewWriter(outputFile)
   defer gzipWriter.Close()

   fmt.Println("Compressing the file.")
   _, err = io.Copy(gzipWriter, inputFile)
   if err != nil {
      panic(err)
   }

   fmt.Println("File compressed successfully.")
}
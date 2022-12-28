//
// Copyright (c) 2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	grpc_client "github.com/openvinotoolkit/model_sever/client/go/kserve-api/grpc-client"

	"google.golang.org/grpc"
)

type Flags struct {
	ModelName    string
	ModelVersion string
	URL          string
}

func parseFlags() Flags {
	var flags Flags
	flag.StringVar(&flags.ModelName, "n", "dummy", "Name of model being served. ")
	flag.StringVar(&flags.ModelVersion, "v", "", "Version of model. ")
	flag.StringVar(&flags.URL, "u", "localhost:9000", "Inference Server URL. ")
	flag.Parse()
	return flags
}

func ModelReadyRequest(client grpc_client.GRPCInferenceServiceClient, modelName string, modelVersion string) *grpc_client.ModelReadyResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	modelReadyRequest := grpc_client.ModelReadyRequest{
		Name:    modelName,
		Version: modelVersion,
	}
	// Submit ServerLive request to server
	modelReadyResponse, err := client.ModelReady(ctx, &modelReadyRequest)
	if err != nil {
		log.Fatalf("Couldn't get model ready: %v", err)
	}
	return modelReadyResponse
}

func main() {
	FLAGS := parseFlags()

	// Connect to gRPC server
	conn, err := grpc.Dial(FLAGS.URL, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Couldn't connect to endpoint %s: %v", FLAGS.URL, err)
	}
	defer conn.Close()

	// Create client from gRPC server connection
	client := grpc_client.NewGRPCInferenceServiceClient(conn)

	modelReadyResponse := ModelReadyRequest(client, FLAGS.ModelName, FLAGS.ModelVersion)
	fmt.Printf("Model Ready: %v\n", modelReadyResponse.Ready)
}
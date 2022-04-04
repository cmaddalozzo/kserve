/*
Copyright 2022 The KServe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1beta1

import (
	"github.com/kserve/kserve/pkg/constants"
	ginkgo "github.com/onsi/ginkgo/v2"
	gomega "github.com/onsi/gomega"
	"testing"
)

var _ = ginkgo.Describe("Inference Service ConfigMap", func() {
	ginkgo.Describe("Autoscaler config", func() {
		ginkgo.Describe("parsing", func() {
			ginkgo.It("Should fail when the JSON is invalid", func() {
				_, err := parseAutoscalerConfig(map[string]string{
					"autoscaler": "{\"defaultAutoscaler\":1}",
				})
				gomega.Expect(err).To(gomega.HaveOccurred())
			})
			ginkgo.It("Should fail when the default autoscaler is invalid", func() {
				_, err := parseAutoscalerConfig(map[string]string{
					"autoscaler": "{\"defaultAutoscaler\":\"bad\"}",
				})
				gomega.Expect(err).To(gomega.HaveOccurred())
			})
			ginkgo.It("Should fail when the enabled autoscalers contains an invalid value", func() {
				_, err := parseAutoscalerConfig(map[string]string{
					"autoscaler": "{\"enabledAutoscalers\":[\"bad\"]}",
				})
				gomega.Expect(err).To(gomega.MatchError("Invalid autoscaler class in enabled autoscalers: bad"))
			})
			ginkgo.It("Should return defaults when the autoscaler entry is missing", func() {
				asConfig, err := parseAutoscalerConfig(map[string]string{})
				gomega.Expect(err).NotTo(gomega.HaveOccurred())
				gomega.Expect(*asConfig).To(gomega.Equal(AutoscalerConfig{
					DefaultAutoscaler: constants.DefaultAutoscalerClass,
					EnabledAutoscalers: []constants.AutoscalerClassType{
						constants.AutoscalerClassHPA,
						constants.AutoscalerClassNone,
					},
				}))
			})
			ginkgo.It("Should fail if default autoscaler is not enabled", func() {
				_, err := parseAutoscalerConfig(map[string]string{
					"autoscaler": "{\"defaultAutoscaler\":\"keda\"}",
				})
				gomega.Expect(err).To(gomega.MatchError("Default autoscaler does not appear in the enabled autoscalers list: keda"))
			})
		})
	})
})

func TestInferenceServiceConfigMap(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t, "Inference Service ConfigMap Suite")
}

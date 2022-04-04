/*
Copyright 2021 The KServe Authors.
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

package keda

import (
	"context"

	kedav1alpha1 "github.com/kedacore/keda/v2/apis/keda/v1alpha1"
	"github.com/kserve/kserve/pkg/apis/serving/v1beta1"
	"github.com/kserve/kserve/pkg/constants"
	"k8s.io/apimachinery/pkg/api/equality"
	apierr "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
)

var log = logf.Log.WithName("KedaReconciler")

type KedaReconciler struct {
	client       client.Client
	scheme       *runtime.Scheme
	ScaledObject *kedav1alpha1.ScaledObject
	componentExt *v1beta1.ComponentExtensionSpec
}

func NewKedaReconciler(client client.Client,
	scheme *runtime.Scheme,
	componentMeta metav1.ObjectMeta,
	componentExt *v1beta1.ComponentExtensionSpec) *KedaReconciler {
	return &KedaReconciler{
		client:       client,
		scheme:       scheme,
		ScaledObject: createScaledObject(componentMeta, componentExt),
		componentExt: componentExt,
	}
}

func createScaledObject(componentMeta metav1.ObjectMeta,
	componentExt *v1beta1.ComponentExtensionSpec) *kedav1alpha1.ScaledObject {
	var minReplicas int32
	if componentExt.MinReplicas == nil || (*componentExt.MinReplicas) < constants.DefaultMinReplicas {
		minReplicas = int32(constants.DefaultMinReplicas)
	} else {
		minReplicas = int32(*componentExt.MinReplicas)
	}

	maxReplicas := int32(componentExt.MaxReplicas)
	if maxReplicas < minReplicas {
		maxReplicas = minReplicas
	}
	so := &kedav1alpha1.ScaledObject{
		ObjectMeta: componentMeta,
		Spec: kedav1alpha1.ScaledObjectSpec{
			ScaleTargetRef: &kedav1alpha1.ScaleTarget{
				Name: componentMeta.Name,
				Kind: "deployment",
			},
			MinReplicaCount: &minReplicas,
			MaxReplicaCount: &maxReplicas,
			Triggers:        componentExt.ScaleTriggers,
		},
	}
	return so
}

//checkHPAExist checks if the hpa exists?
func (r *KedaReconciler) checkScaledObjectExists(client client.Client) (constants.CheckResultType, *kedav1alpha1.ScaledObject, error) {
	//get hpa
	existingScaledObject := &kedav1alpha1.ScaledObject{}
	err := client.Get(context.TODO(), types.NamespacedName{
		Namespace: r.ScaledObject.ObjectMeta.Namespace,
		Name:      r.ScaledObject.ObjectMeta.Name,
	}, existingScaledObject)
	if err != nil {
		if apierr.IsNotFound(err) {
			return constants.CheckResultCreate, nil, nil
		}
		return constants.CheckResultUnknown, nil, err
	}

	//existed, check equivalent
	if semanticScaledObjectEquals(r.ScaledObject, existingScaledObject) {
		return constants.CheckResultExisted, existingScaledObject, nil
	}
	return constants.CheckResultUpdate, existingScaledObject, nil
}

func semanticScaledObjectEquals(desired, existing *kedav1alpha1.ScaledObject) bool {
	return equality.Semantic.DeepEqual(desired.Spec.MaxReplicaCount, *existing.Spec.MaxReplicaCount) &&
		equality.Semantic.DeepEqual(*desired.Spec.MinReplicaCount, *existing.Spec.MinReplicaCount) &&
		equality.Semantic.DeepEqual(desired.Spec.Triggers, existing.Spec.Triggers)
}

// Set controller references on any resources we created
func (r *KedaReconciler) SetControllerReferences(owner metav1.Object, scheme *runtime.Scheme) error {
	return controllerutil.SetControllerReference(owner, r.ScaledObject, scheme)
}

//Reconcile ...
func (r *KedaReconciler) Reconcile() error {
	//reconcile Service
	checkResult, existing, err := r.checkScaledObjectExists(r.client)
	if err != nil {
		return err
	}

	if checkResult == constants.CheckResultCreate {
		return r.client.Create(context.TODO(), r.ScaledObject)
	} else if checkResult == constants.CheckResultUpdate {
		newScaledObject := r.ScaledObject.DeepCopy()
		newScaledObject.ObjectMeta.ResourceVersion = existing.ObjectMeta.ResourceVersion
		return r.client.Update(context.TODO(), newScaledObject)
	}
	return nil
}

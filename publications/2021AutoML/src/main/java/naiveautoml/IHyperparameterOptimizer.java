package naiveautoml;

import java.util.Collection;

import org.api4.java.algorithm.Timeout;
import org.api4.java.common.attributedobjects.IObjectEvaluator;

import ai.libs.jaicore.components.api.IComponentInstance;

public interface IHyperparameterOptimizer {

	public Collection<IComponentInstance> optimize(IComponentInstance basis, IObjectEvaluator<IComponentInstance, Double> benchmark, Timeout timeout);
}

package org.zsyu.ml.gd;

import org.zsyu.ml.gd.ASGD;
import org.zsyu.ml.gd.GD;
import org.zsyu.ml.gd.SGD;

import junit.framework.TestCase;

public class TestASGD extends TestCase {
	
	public void testASGD() throws Exception {
		
		// 生成训练数据  y = a + b * x1 + c * x2
		int dataSize = 100;
		double[][] x_data = new double[dataSize][3];
		double[] y_data = new double[dataSize];
		double x1 = 0.0;
		double x2 = 0.0;
		for (int i = 0; i < dataSize; i++) {
			x_data[i][0] = 1.0;
			x1 = Math.random();
			x2 = Math.random();
			x_data[i][1] = x1;
			x_data[i][2] = x2;
			y_data[i] = 0.2 + 0.3 * x1 + 0.5 * x2;
			
		}
		
		// 生成训练数据 y = a + b * x1
//		int dataSize = 100;
//		double[][] x_data = new double[dataSize][2];
//		double[] y_data = new double[dataSize];
//		double x1 = 0.0;
//		for (int i = 0; i < dataSize; i++) {
//			x_data[i][0] = 1.0;
//			x1 = Math.random();
//			x_data[i][1] = x1;
//			y_data[i] = 0.2 + 0.3 * x1;
//			
//		}
		
		//训练速率
		double alpha = 0.5; 
		double deviation = 0.000001;
		//每次更新参数使用样本大小
		int batchSize = 20;
		//迭代次数上限
		int iterations = 10000;
		
		//梯度下降法，如果batchSize的大小等于样本总数，则为批量梯度下降法
		GD.gdTrain(x_data, y_data, alpha, deviation, batchSize, iterations);
		
		//随机梯度下降法
		//随机梯度下降法中，batchSize 不应该等于样本总数，如果等，将退化为批量梯度下降
		/**
		 * 随机策略一
		 * 每一轮迭代样本前，生成0-（n-1）（n为样本总数）的n个数组成的数组，并将其随机打乱,
		 * 在根据打乱的数组按batchSize的大小取样本进行训练；
		 * 该方法中，每一轮中的每一次更新参数的样本是随机的，但每一轮中一个样本只使用一次，即每一轮中，每一次更新参数的样本不会重复使用，
		 * 也就是说没一轮下来将用完所有的样本
		 * 但有可能出现一轮未完成就已经收敛
		 * 
		 * 如果batchSize = 1 为SGD
		 * 如果batchSize > 1 为MSGD
		 */
		SGD.sgdTrain(x_data, y_data, alpha, deviation, batchSize, iterations);
		
		/**
		 * 随机策略二
		 * 每次更新参数是从样本n中随机选取batchSize个样本进行训练
		 * 当前次和下一次更新参数的样本可能存在同一个或几个样本，
		 * 迭代次数少时，有的样本可能未被使用，
		 * 当迭代次数足够多时，应该所有的样本都已使用到
		 * 
		 * 如果batchSize = 1 为SGD
		 * 如果batchSize > 1 为MSGD
		 */
		SGD.sgdTrain2(x_data, y_data, alpha, deviation, batchSize, iterations);
		
		//平均随机梯度下降
		/**
		 *  随机策略一
		 */
		ASGD.asgdTrain(x_data, y_data, alpha, deviation, batchSize, iterations);
		
		/**
		 *  随机策略二
		 */
		ASGD.asgdTrain2(x_data, y_data, alpha, deviation, batchSize, iterations);
		
		
		
	}

}

using System;
using System.Diagnostics;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using Random = UnityEngine.Random;
using Debug = UnityEngine.Debug;

namespace GpuMatMul.Test
{

    public class Test : MonoBehaviour {

        [SerializeField, Range(8, 2048)] protected int a = 1024, b = 512, c = 512;
        [SerializeField] protected ComputeShader matmul;

        void Start () {
            float[,] A = new float[a, b];
            float[,] B = new float[b, c];

            for(int y = 0; y < a; y++)
            {
                for(int x = 0; x < b; x++)
                {
                    A[y, x] = x;
                }
            }

            for(int y = 0; y < b; y++)
            {
                for(int x = 0; x < c; x++)
                {
                    B[y, x] = x;
                }
            }

            /*
            if(!GPUMatrixMultiplication.Test(matmul, A, B))
            {
                Debug.LogWarning("CPU impl & GPU impl results are not same.");
            }
            */

            const int iterations = 32;

            Measure("SharedMemory method", () => {
                GPUMatrixMultiplication.Multiply(matmul, A, B, GPUMatrixMultiplicationMethod.SharedMemory);
            }, iterations);

            Measure("Naive method", () => {
                GPUMatrixMultiplication.Multiply(matmul, A, B, GPUMatrixMultiplicationMethod.Naive);
            }, iterations);

        }

        void Measure(string label, Action act, int iterations)
        {
            GC.Collect();

            // run once outside of loop to avoid initialization costs
            act.Invoke();
            Stopwatch sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                act.Invoke();
            }
            sw.Stop();

            Debug.Log(label + " : " + (sw.ElapsedMilliseconds / iterations).ToString());
        }

    }

}



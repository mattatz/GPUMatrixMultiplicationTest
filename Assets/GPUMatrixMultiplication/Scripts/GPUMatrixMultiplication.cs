using System.Runtime.InteropServices;

using UnityEngine;

namespace GpuMatMul
{

    public enum GPUMatrixMultiplicationMethod
    {
        SharedMemory = 1,
        Naive = 2
    };

    public class GPUMatrixMultiplication {

        public static float[, ] Multiply(ComputeShader matmul, float[,] A, float[,] B, GPUMatrixMultiplicationMethod method = GPUMatrixMultiplicationMethod.SharedMemory)
        {
            switch(method)
            {
                case GPUMatrixMultiplicationMethod.SharedMemory:
                    return Multiply(matmul, matmul.FindKernel("MatMul"), A, B);

                case GPUMatrixMultiplicationMethod.Naive:
                    return Multiply(matmul, matmul.FindKernel("MatMulNaive"), A, B);

                default:
                    return Multiply(matmul, matmul.FindKernel("MatMul"), A, B);
            }
        }

        protected static float[, ] Multiply(ComputeShader matmul, int kernel, float[,] A, float[,] B)
        {
            int aRows = A.GetLength(0);
            int aCols = A.GetLength(1);
            int bRows = B.GetLength(0);
            int bCols = B.GetLength(1);
            int cRows = aRows, cCols = bCols;

            if(aCols != bRows)
            {
                throw new System.Exception("mat A & B dimensions does not match");
            }

            var ABuf = new ComputeBuffer(aRows * aCols, Marshal.SizeOf(typeof(float)));
            var BBuf = new ComputeBuffer(bRows * bCols, Marshal.SizeOf(typeof(float)));
            var CBuf = new ComputeBuffer(aRows * bCols, Marshal.SizeOf(typeof(float)));

            ABuf.SetData(A);
            BBuf.SetData(B);

            matmul.SetBuffer(kernel, "_A", ABuf);
            matmul.SetBuffer(kernel, "_B", BBuf);
            matmul.SetBuffer(kernel, "_C", CBuf);

            uint tx, ty, tz;
            matmul.GetKernelThreadGroupSizes(kernel, out tx, out ty, out tz);

            matmul.SetInt("_ARows", aRows); matmul.SetInt("_ACols", aCols);
            matmul.SetInt("_BRows", bRows); matmul.SetInt("_BCols", bCols);
            matmul.SetInt("_CRows", cRows); matmul.SetInt("_CCols", cCols);

            matmul.Dispatch(kernel, Mathf.FloorToInt(((int)cCols - 1) / tx) + 1, Mathf.FloorToInt(((int)cRows - 1) / ty) + 1, (int)tz);

            float[,] C = new float[aRows, bCols];
            CBuf.GetData(C);

            ABuf.Release();
            BBuf.Release();
            CBuf.Release();

            return C;
        }

        public static bool Test(ComputeShader matmul, float[,] A, float[,] B)
        {
            var C0 = MultiplyCPUTest(A, B);
            var C1 = Multiply(matmul, A, B);

            Debug.Log(ToString(C0));
            Debug.Log(ToString(C1));

            int rows = C0.GetLength(0);
            int columns = C0.GetLength(1);
            for(int y = 0; y < rows; y++)
            {
                for(int x = 0; x < columns; x++)
                {
                    if(!Mathf.Approximately(C0[y, x], C1[y, x]))
                    {
                        Debug.Log(C0[y, x] + " - " + C1[y, x]);
                        return false;
                    }
                }
            }
            return true;
        }

        protected static string ToString(float[,] M)
        {
            int dim0 = M.GetLength(0);
            int dim1 = M.GetLength(1);

            var rows = new string[dim0];
            for(int i = 0; i < dim0; i++)
            {
                var row = new string[dim1];
                for(int j = 0; j < dim1; j++)
                {
                    var v = M[i, j];
                    row[j] = v.ToString();
                }
                rows[i] = "[" + string.Join(",", row) + "]";
            }
            return string.Join("\n", rows);
        }

        protected static float[, ] MultiplyCPUTest(float[,] A, float[,] B)
        {
            int dim0 = A.GetLength(0);
            int dim1 = A.GetLength(1);
            int dim2 = B.GetLength(0);
            int dim3 = B.GetLength(1);

            if(dim1 != dim2)
            {
                throw new System.Exception("mat A & B dimensions does not match");
            }

            float[,] C = new float[dim0, dim3];

            for(int i = 0; i < dim0; i++)
            {
                for(int j = 0; j < dim3; j++)
                {
                    var acc = 0f;
                    for(int k = 0; k < dim1; k++)
                    {
                        acc += A[i, k] * B[k, j];
                    }
                    C[i, j] = acc;
                }
            }

            return C;
        }


    }

}



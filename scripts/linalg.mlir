#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6, d0, d2 + d3, d4 + d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d0, d3, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6, d1, d2, d4)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1)>
module @IrToHlo.22 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<16xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<16x7200xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<8xf32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<8x3x3x3xf32> {mhlo.sharding = "{replicated}"}, %arg4: tensor<16x3x32x32xf32> {mhlo.sharding = "{replicated}"}) -> tensor<16x16xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %0 = tensor.empty() : tensor<8x3x3x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<8x3x3x3xf32>) outs(%0 : tensor<8x3x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x3x3x3xf32>
    %2 = tensor.empty() : tensor<16x8x30x30xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<16x8x30x30xf32>) -> tensor<16x8x30x30xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["reduction", "parallel", "parallel", "reduction", "parallel", "reduction", "parallel"]} ins(%arg4, %1 : tensor<16x3x32x32xf32>, tensor<8x3x3x3xf32>) outs(%3 : tensor<16x8x30x30xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %20 = arith.mulf %in, %in_4 : f32
      %21 = arith.addf %out, %20 : f32
      linalg.yield %21 : f32
    } -> tensor<16x8x30x30xf32>
    %collapsed = tensor.collapse_shape %4 [[0, 1, 2, 3]] : tensor<16x8x30x30xf32> into tensor<115200xf32>
    %cast = tensor.cast %collapsed : tensor<115200xf32> to tensor<115200xf32>
    %expanded = tensor.expand_shape %cast [[0, 1, 2, 3]] output_shape [16, 8, 30, 30] : tensor<115200xf32> into tensor<16x8x30x30xf32>
    %5 = tensor.empty() : tensor<16x8x30x30xf32>
    %6 = linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<8xf32>) outs(%5 : tensor<16x8x30x30xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x8x30x30xf32>
    %7 = tensor.empty() : tensor<16x8x30x30xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded, %6 : tensor<16x8x30x30xf32>, tensor<16x8x30x30xf32>) outs(%7 : tensor<16x8x30x30xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %20 = arith.addf %in, %in_4 : f32
      linalg.yield %20 : f32
    } -> tensor<16x8x30x30xf32>
    %collapsed_1 = tensor.collapse_shape %8 [[0], [1, 2, 3]] : tensor<16x8x30x30xf32> into tensor<16x7200xf32>
    %9 = tensor.empty() : tensor<7200x16xf32>
    %10 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<16x7200xf32>) outs(%9 : tensor<7200x16xf32>) attrs =  {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[7200,16]{0,1}"} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<7200x16xf32>
    %11 = tensor.empty() : tensor<16x16xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %12 = linalg.fill ins(%cst_2 : f32) outs(%11 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %13 = linalg.matmul ins(%collapsed_1, %10 : tensor<16x7200xf32>, tensor<7200x16xf32>) outs(%12 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %14 = tensor.empty() : tensor<16x16xf32>
    %15 = linalg.generic {indexing_maps = [#map7, #map6], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<16xf32>) outs(%14 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x16xf32>
    %16 = tensor.empty() : tensor<16x16xf32>
    %17 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%13, %15 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%16 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %20 = arith.addf %in, %in_4 : f32
      linalg.yield %20 : f32
    } -> tensor<16x16xf32>
    %18 = tensor.empty() : tensor<16x16xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %19 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<16x16xf32>) outs(%18 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      %20 = arith.maximumf %in, %cst_3 : f32
      linalg.yield %20 : f32
    } -> tensor<16x16xf32>
    return %19 : tensor<16x16xf32>
  }
}


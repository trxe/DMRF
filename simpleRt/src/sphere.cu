#include "sphere.cuh"
#include "hittable.cuh"
#include "aabb.cuh"
#include "vec3.cuh"
#include "material.cuh"

__host__ __device__ sphere::sphere(vec3 cen, float r, material *m)
	: radius(r){
	type = type_sphere;
	center = cen;
	mat_ptr = m;
}

__host__ __device__ bool sphere::bounding_box(float t0, float t1, aabb& box){
	box = aabb(center - vec3(radius, radius, radius),
		center + vec3(radius, radius, radius));
	return true;
}

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3 oc = r.origin() - center;

	// uint32_t pixel_index = threadIdx.x + blockIdx.x * blockDim.x;
	// if (pixel_index == 0) printf("center of %d: %f %f %f\n", pixel_index, oc[0], oc[1], oc[2]);

	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}

__device__ vec3 sphere::sample(curandState *local_rand_state) const {
  vec3 p;
  do {
    p = 2.0f*vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state)) - vec3(1,1,1);
  } while (p.length() < 0.000001f);
  p = p / p.length();
  return center + radius * p;
}

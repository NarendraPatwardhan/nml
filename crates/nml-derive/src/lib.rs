//! Derives NML's single structural model mapping.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DataEnum, DataStruct, DeriveInput, Fields, GenericArgument, Index,
    PathArguments, Type, parse_macro_input,
};

#[proc_macro_derive(ParameterTree, attributes(nml))]
pub fn derive_parameter_tree(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand(input: DeriveInput) -> syn::Result<TokenStream2> {
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            input.generics,
            "ParameterTree derive does not infer generic model mappings; implement ParameterTree manually",
        ));
    }
    let name = input.ident;
    let loaded = format_ident!("__{}Loaded", name);
    let visibility = input.vis;
    let (loaded_definition, visit_parameters, visit_loaded, load_parameters) = match input.data {
        Data::Struct(data) => expand_struct(&name, &loaded, &visibility, data)?,
        Data::Enum(data) => expand_enum(&name, &loaded, &visibility, data)?,
        Data::Union(union) => {
            return Err(syn::Error::new_spanned(
                union.union_token,
                "ParameterTree cannot derive an ownership-safe mapping for unions",
            ));
        }
    };
    Ok(quote! {
        #[doc(hidden)]
        #loaded_definition

        impl ::nml::ParameterTree for #name {
            type Loaded = #loaded;

            fn visit_parameters(
                &self,
                __prefix: &str,
                __visitor: &mut dyn FnMut(&str, &::nml::Parameter),
            ) {
                #visit_parameters
            }

            fn visit_loaded(
                __loaded: &Self::Loaded,
                __prefix: &str,
                __visitor: &mut dyn FnMut(&str, &::nml::LoadedParameter),
            ) {
                #visit_loaded
            }

            fn load_parameters<__NmlError>(
                &self,
                __prefix: &str,
                __resolve: &mut impl FnMut(&str, &::nml::Parameter) -> ::std::result::Result<::nml::LoadedParameter, __NmlError>,
            ) -> ::std::result::Result<Self::Loaded, __NmlError> {
                #load_parameters
            }
        }
    })
}

fn expand_struct(
    _name: &syn::Ident,
    loaded: &syn::Ident,
    visibility: &syn::Visibility,
    data: DataStruct,
) -> syn::Result<(TokenStream2, TokenStream2, TokenStream2, TokenStream2)> {
    match data.fields {
        Fields::Named(fields) => {
            let kept = fields
                .named
                .iter()
                .filter(|field| !is_skipped(&field.attrs))
                .collect::<Vec<_>>();
            let definitions = kept.iter().map(|field| {
                let visibility = &field.vis;
                let ident = field.ident.as_ref().expect("named field");
                let ty = loaded_type(&field.ty);
                quote!(#visibility #ident: #ty)
            });
            let parameter_visits = kept.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                visit_parameter_tokens(
                    quote!(&self.#ident),
                    &field.ty,
                    quote!(__prefix),
                    ident.to_string(),
                )
            });
            let loaded_visits = kept.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                visit_loaded_tokens(
                    quote!(&__loaded.#ident),
                    &field.ty,
                    quote!(__prefix),
                    ident.to_string(),
                )
            });
            let initializers = kept.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                let value = load_parameters_tokens(
                    quote!(&self.#ident),
                    &field.ty,
                    quote!(__prefix),
                    ident.to_string(),
                );
                quote!(#ident: #value)
            });
            Ok((
                quote!(#visibility struct #loaded { #(#definitions,)* }),
                quote!(#(#parameter_visits)*),
                quote!(#(#loaded_visits)*),
                quote!(Ok(#loaded { #(#initializers,)* })),
            ))
        }
        Fields::Unnamed(fields) => {
            let kept = fields
                .unnamed
                .iter()
                .enumerate()
                .filter(|(_, field)| !is_skipped(&field.attrs))
                .collect::<Vec<_>>();
            let definitions = kept.iter().map(|(_, field)| {
                let visibility = &field.vis;
                let ty = loaded_type(&field.ty);
                quote!(#visibility #ty)
            });
            let parameter_visits = kept.iter().map(|(index, field)| {
                let index = Index::from(*index);
                visit_parameter_tokens(
                    quote!(&self.#index),
                    &field.ty,
                    quote!(__prefix),
                    index.index.to_string(),
                )
            });
            let loaded_visits = kept.iter().enumerate().map(|(mapped, (_, field))| {
                let index = Index::from(mapped);
                visit_loaded_tokens(
                    quote!(&__loaded.#index),
                    &field.ty,
                    quote!(__prefix),
                    index.index.to_string(),
                )
            });
            let initializers = kept.iter().map(|(index, field)| {
                let index = Index::from(*index);
                load_parameters_tokens(
                    quote!(&self.#index),
                    &field.ty,
                    quote!(__prefix),
                    index.index.to_string(),
                )
            });
            Ok((
                quote!(#visibility struct #loaded(#(#definitions,)*);),
                quote!(#(#parameter_visits)*),
                quote!(#(#loaded_visits)*),
                quote!(Ok(#loaded(#(#initializers,)*))),
            ))
        }
        Fields::Unit => Ok((
            quote!(#visibility struct #loaded;),
            TokenStream2::new(),
            TokenStream2::new(),
            quote!(Ok(#loaded)),
        )),
    }
}

fn expand_enum(
    name: &syn::Ident,
    loaded: &syn::Ident,
    visibility: &syn::Visibility,
    data: DataEnum,
) -> syn::Result<(TokenStream2, TokenStream2, TokenStream2, TokenStream2)> {
    let variants = data.variants.iter().map(|variant| {
        let variant_name = &variant.ident;
        match &variant.fields {
            Fields::Named(fields) => {
                let definitions = fields
                    .named
                    .iter()
                    .filter(|field| !is_skipped(&field.attrs))
                    .map(|field| {
                        let ident = field.ident.as_ref().expect("named field");
                        let ty = loaded_type(&field.ty);
                        quote!(#ident: #ty)
                    });
                quote!(#variant_name { #(#definitions,)* })
            }
            Fields::Unnamed(fields) => {
                let definitions = fields
                    .unnamed
                    .iter()
                    .filter(|field| !is_skipped(&field.attrs))
                    .map(|field| loaded_type(&field.ty));
                quote!(#variant_name(#(#definitions,)*))
            }
            Fields::Unit => quote!(#variant_name),
        }
    });
    let parameter_arms = data
        .variants
        .iter()
        .map(|variant| enum_visit_parameter_arm(name, variant));
    let loaded_arms = data
        .variants
        .iter()
        .map(|variant| enum_visit_loaded_arm(loaded, variant));
    let load_parameters_arms = data
        .variants
        .iter()
        .map(|variant| enum_load_parameters_arm(name, loaded, variant));
    Ok((
        quote!(#visibility enum #loaded { #(#variants,)* }),
        quote!(match self { #(#parameter_arms,)* }),
        quote!(match __loaded { #(#loaded_arms,)* }),
        quote!(match self { #(#load_parameters_arms,)* }),
    ))
}

fn enum_visit_parameter_arm(name: &syn::Ident, variant: &syn::Variant) -> TokenStream2 {
    let variant_name = &variant.ident;
    match &variant.fields {
        Fields::Named(fields) => {
            let kept = fields
                .named
                .iter()
                .filter(|field| !is_skipped(&field.attrs))
                .collect::<Vec<_>>();
            let bindings = kept
                .iter()
                .map(|field| field.ident.as_ref().expect("named field"));
            let visits = kept.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                visit_parameter_tokens(
                    quote!(#ident),
                    &field.ty,
                    quote!(__prefix),
                    ident.to_string(),
                )
            });
            quote!(#name::#variant_name { #(#bindings,)* .. } => { #(#visits)* })
        }
        Fields::Unnamed(fields) => {
            let all = fields
                .unnamed
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    if is_skipped(&field.attrs) {
                        format_ident!("_skip_{index}")
                    } else {
                        format_ident!("field_{index}")
                    }
                })
                .collect::<Vec<_>>();
            let visits = fields
                .unnamed
                .iter()
                .enumerate()
                .filter(|(_, field)| !is_skipped(&field.attrs))
                .map(|(index, field)| {
                    let binding = &all[index];
                    visit_parameter_tokens(
                        quote!(#binding),
                        &field.ty,
                        quote!(__prefix),
                        index.to_string(),
                    )
                });
            quote!(#name::#variant_name(#(#all,)*) => { #(#visits)* })
        }
        Fields::Unit => quote!(#name::#variant_name => {}),
    }
}

fn enum_visit_loaded_arm(loaded: &syn::Ident, variant: &syn::Variant) -> TokenStream2 {
    let variant_name = &variant.ident;
    match &variant.fields {
        Fields::Named(fields) => {
            let kept = fields
                .named
                .iter()
                .filter(|field| !is_skipped(&field.attrs))
                .collect::<Vec<_>>();
            let bindings = kept
                .iter()
                .map(|field| field.ident.as_ref().expect("named field"));
            let visits = kept.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                visit_loaded_tokens(
                    quote!(#ident),
                    &field.ty,
                    quote!(__prefix),
                    ident.to_string(),
                )
            });
            quote!(#loaded::#variant_name { #(#bindings,)* } => { #(#visits)* })
        }
        Fields::Unnamed(fields) => {
            let kept = fields
                .unnamed
                .iter()
                .enumerate()
                .filter(|(_, field)| !is_skipped(&field.attrs))
                .collect::<Vec<_>>();
            let bindings = kept
                .iter()
                .enumerate()
                .map(|(mapped, _)| format_ident!("field_{mapped}"))
                .collect::<Vec<_>>();
            let visits = kept.iter().enumerate().map(|(mapped, (original, field))| {
                let binding = &bindings[mapped];
                visit_loaded_tokens(
                    quote!(#binding),
                    &field.ty,
                    quote!(__prefix),
                    original.to_string(),
                )
            });
            quote!(#loaded::#variant_name(#(#bindings,)*) => { #(#visits)* })
        }
        Fields::Unit => quote!(#loaded::#variant_name => {}),
    }
}

fn enum_load_parameters_arm(
    name: &syn::Ident,
    loaded: &syn::Ident,
    variant: &syn::Variant,
) -> TokenStream2 {
    let variant_name = &variant.ident;
    match &variant.fields {
        Fields::Named(fields) => {
            let kept = fields
                .named
                .iter()
                .filter(|field| !is_skipped(&field.attrs))
                .collect::<Vec<_>>();
            let bindings = kept
                .iter()
                .map(|field| field.ident.as_ref().expect("named field"));
            let values = kept.iter().map(|field| {
                let ident = field.ident.as_ref().expect("named field");
                let value = load_parameters_tokens(
                    quote!(#ident),
                    &field.ty,
                    quote!(__prefix),
                    ident.to_string(),
                );
                quote!(#ident: #value)
            });
            quote!(#name::#variant_name { #(#bindings,)* .. } => Ok(#loaded::#variant_name { #(#values,)* }))
        }
        Fields::Unnamed(fields) => {
            let all = fields
                .unnamed
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    if is_skipped(&field.attrs) {
                        format_ident!("_skip_{index}")
                    } else {
                        format_ident!("field_{index}")
                    }
                })
                .collect::<Vec<_>>();
            let values = fields
                .unnamed
                .iter()
                .enumerate()
                .filter(|(_, field)| !is_skipped(&field.attrs))
                .map(|(index, field)| {
                    let binding = &all[index];
                    load_parameters_tokens(
                        quote!(#binding),
                        &field.ty,
                        quote!(__prefix),
                        index.to_string(),
                    )
                });
            quote!(#name::#variant_name(#(#all,)*) => Ok(#loaded::#variant_name(#(#values,)*)))
        }
        Fields::Unit => quote!(#name::#variant_name => Ok(#loaded::#variant_name)),
    }
}

fn is_skipped(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        attribute.path().is_ident("nml")
            && attribute
                .parse_nested_meta(|meta| {
                    if meta.path.is_ident("skip") {
                        Ok(())
                    } else {
                        Err(meta.error("expected `skip`"))
                    }
                })
                .is_ok()
    })
}

fn loaded_type(ty: &Type) -> TokenStream2 {
    if is_parameter(ty) {
        return quote!(::nml::LoadedParameter);
    }
    if let Some((container, arguments)) = container(ty) {
        return match container.as_str() {
            "Option" => {
                let inner = loaded_type(arguments[0]);
                quote!(Option<#inner>)
            }
            "Vec" => {
                let inner = loaded_type(arguments[0]);
                quote!(Vec<#inner>)
            }
            "Box" => {
                let inner = loaded_type(arguments[0]);
                quote!(Box<#inner>)
            }
            _ => unreachable!(),
        };
    }
    match ty {
        Type::Array(array) => {
            let inner = loaded_type(&array.elem);
            let length = &array.len;
            quote!([#inner; #length])
        }
        Type::Tuple(tuple) => {
            let elements = tuple.elems.iter().map(loaded_type);
            quote!((#(#elements,)*))
        }
        _ => quote!(<#ty as ::nml::ParameterTree>::Loaded),
    }
}

fn visit_parameter_tokens(
    value: TokenStream2,
    ty: &Type,
    prefix: TokenStream2,
    segment: String,
) -> TokenStream2 {
    let path = path_tokens(prefix.clone(), &segment);
    if is_parameter(ty) {
        return quote!({ let __path = #path; __visitor(&__path, #value); });
    }
    if let Some((container, arguments)) = container(ty) {
        let inner = arguments[0];
        return match container.as_str() {
            "Option" => {
                let nested =
                    visit_parameter_tokens(quote!(__value), inner, quote!(&__path), String::new());
                quote!({ let __path = #path; if let Some(__value) = #value { #nested } })
            }
            "Vec" => {
                let nested = visit_parameter_tokens(
                    quote!(__value),
                    inner,
                    quote!(&__item_path),
                    String::new(),
                );
                quote!({ let __path = #path; for (__index, __value) in (#value).iter().enumerate() { let __item_path = if __path.is_empty() { __index.to_string() } else { format!("{}.{__index}", __path) }; #nested } })
            }
            "Box" => visit_parameter_tokens(quote!(&**#value), inner, prefix, segment),
            _ => unreachable!(),
        };
    }
    match ty {
        Type::Array(array) => {
            let nested = visit_parameter_tokens(
                quote!(__value),
                &array.elem,
                quote!(&__item_path),
                String::new(),
            );
            quote!({ let __path = #path; for (__index, __value) in (#value).iter().enumerate() { let __item_path = if __path.is_empty() { __index.to_string() } else { format!("{}.{__index}", __path) }; #nested } })
        }
        Type::Tuple(tuple) => {
            let visits = tuple.elems.iter().enumerate().map(|(index, field)| {
                let index = Index::from(index);
                visit_parameter_tokens(
                    quote!(&(#value).#index),
                    field,
                    quote!(&__path),
                    index.index.to_string(),
                )
            });
            quote!({ let __path = #path; #(#visits)* })
        }
        _ => {
            quote!({ let __path = #path; ::nml::ParameterTree::visit_parameters(#value, &__path, __visitor); })
        }
    }
}

fn visit_loaded_tokens(
    value: TokenStream2,
    ty: &Type,
    prefix: TokenStream2,
    segment: String,
) -> TokenStream2 {
    let path = path_tokens(prefix.clone(), &segment);
    if is_parameter(ty) {
        return quote!({ let __path = #path; __visitor(&__path, #value); });
    }
    if let Some((container, arguments)) = container(ty) {
        let inner = arguments[0];
        return match container.as_str() {
            "Option" => {
                let nested =
                    visit_loaded_tokens(quote!(__value), inner, quote!(&__path), String::new());
                quote!({ let __path = #path; if let Some(__value) = #value { #nested } })
            }
            "Vec" => {
                let nested = visit_loaded_tokens(
                    quote!(__value),
                    inner,
                    quote!(&__item_path),
                    String::new(),
                );
                quote!({ let __path = #path; for (__index, __value) in (#value).iter().enumerate() { let __item_path = if __path.is_empty() { __index.to_string() } else { format!("{}.{__index}", __path) }; #nested } })
            }
            "Box" => visit_loaded_tokens(quote!(&**#value), inner, prefix, segment),
            _ => unreachable!(),
        };
    }
    match ty {
        Type::Array(array) => {
            let nested = visit_loaded_tokens(
                quote!(__value),
                &array.elem,
                quote!(&__item_path),
                String::new(),
            );
            quote!({ let __path = #path; for (__index, __value) in (#value).iter().enumerate() { let __item_path = if __path.is_empty() { __index.to_string() } else { format!("{}.{__index}", __path) }; #nested } })
        }
        Type::Tuple(tuple) => {
            let visits = tuple.elems.iter().enumerate().map(|(index, field)| {
                let index = Index::from(index);
                visit_loaded_tokens(
                    quote!(&(#value).#index),
                    field,
                    quote!(&__path),
                    index.index.to_string(),
                )
            });
            quote!({ let __path = #path; #(#visits)* })
        }
        _ => {
            quote!({ let __path = #path; <#ty as ::nml::ParameterTree>::visit_loaded(#value, &__path, __visitor); })
        }
    }
}

fn load_parameters_tokens(
    value: TokenStream2,
    ty: &Type,
    prefix: TokenStream2,
    segment: String,
) -> TokenStream2 {
    let path = path_tokens(prefix.clone(), &segment);
    if is_parameter(ty) {
        return quote!({ let __path = #path; __resolve(&__path, #value)? });
    }
    if let Some((container, arguments)) = container(ty) {
        let inner = arguments[0];
        return match container.as_str() {
            "Option" => {
                let nested =
                    load_parameters_tokens(quote!(__value), inner, quote!(&__path), String::new());
                quote!({ let __path = #path; match #value { Some(__value) => Some(#nested), None => None } })
            }
            "Vec" => {
                let nested = load_parameters_tokens(
                    quote!(__value),
                    inner,
                    quote!(&__item_path),
                    String::new(),
                );
                quote!({ let __path = #path; (#value).iter().enumerate().map(|(__index, __value)| { let __item_path = if __path.is_empty() { __index.to_string() } else { format!("{}.{__index}", __path) }; ::std::result::Result::Ok(#nested) }).collect::<::std::result::Result<::std::vec::Vec<_>, __NmlError>>()? })
            }
            "Box" => {
                let nested = load_parameters_tokens(quote!(&**#value), inner, prefix, segment);
                quote!(Box::new(#nested))
            }
            _ => unreachable!(),
        };
    }
    match ty {
        Type::Array(array) => {
            let nested = load_parameters_tokens(
                quote!(__value),
                &array.elem,
                quote!(&__item_path),
                String::new(),
            );
            quote!({ let __path = #path; let __values = (#value).iter().enumerate().map(|(__index, __value)| { let __item_path = if __path.is_empty() { __index.to_string() } else { format!("{}.{__index}", __path) }; ::std::result::Result::Ok(#nested) }).collect::<::std::result::Result<::std::vec::Vec<_>, __NmlError>>()?; __values.try_into().ok().expect("loaded array length matches source") })
        }
        Type::Tuple(tuple) => {
            let values = tuple.elems.iter().enumerate().map(|(index, field)| {
                let index = Index::from(index);
                load_parameters_tokens(
                    quote!(&(#value).#index),
                    field,
                    quote!(&__path),
                    index.index.to_string(),
                )
            });
            quote!({ let __path = #path; (#(#values,)*) })
        }
        _ => {
            quote!({ let __path = #path; <#ty as ::nml::ParameterTree>::load_parameters(#value, &__path, __resolve)? })
        }
    }
}

fn is_parameter(ty: &Type) -> bool {
    matches!(ty, Type::Path(path) if path.path.segments.last().is_some_and(|segment| segment.ident == "Parameter"))
}

fn container(ty: &Type) -> Option<(String, Vec<&Type>)> {
    let Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    let name = segment.ident.to_string();
    if !matches!(name.as_str(), "Option" | "Vec" | "Box") {
        return None;
    }
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    let types = arguments
        .args
        .iter()
        .filter_map(|argument| match argument {
            GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect::<Vec<_>>();
    (!types.is_empty()).then_some((name, types))
}

fn path_tokens(prefix: TokenStream2, segment: &str) -> TokenStream2 {
    if segment.is_empty() {
        quote!((#prefix).to_string())
    } else {
        quote!(if (#prefix).is_empty() { #segment.to_owned() } else { format!("{}.{}", #prefix, #segment) })
    }
}
